// Package agent wires a model, a tool registry and saved context into a small
// tool-calling loop.
//
// The loop is deliberately boring: send the conversation, read the reply, run
// any tool calls the reply asked for, feed the results back, repeat until the
// model answers without calling a tool or the turn budget runs out.
package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"regexp"
	"strings"

	"micro-pa/internal/llm"
	"micro-pa/internal/tools"
)

// DefaultMaxTurns bounds how many tool rounds one question may use.
const DefaultMaxTurns = 8

// Call is one tool call parsed out of a model reply.
type Call struct {
	Name      string
	Arguments map[string]any
}

var (
	toolCallBlock = regexp.MustCompile(`(?s)<tool_call>\s*(.*?)\s*</tool_call>`)
	jsonFence     = regexp.MustCompile("(?s)```(?:json)?\\s*(\\{.*?\\})\\s*```")
)

// ParseToolCalls extracts tool calls from a reply. It prefers the explicit
// <tool_call> form and falls back to a fenced JSON object, which is what some
// models produce when they ignore the instructions.
func ParseToolCalls(reply string) []Call {
	var calls []Call
	for _, match := range toolCallBlock.FindAllStringSubmatch(reply, -1) {
		if call, ok := parseCall([]byte(match[1])); ok {
			calls = append(calls, call)
		}
	}
	if len(calls) > 0 {
		return calls
	}
	for _, match := range jsonFence.FindAllStringSubmatch(reply, -1) {
		if call, ok := parseCall([]byte(match[1])); ok {
			calls = append(calls, call)
		}
	}
	if len(calls) > 0 {
		return calls
	}
	// A bare JSON object as the entire reply is also treated as a call, but
	// only when it really looks like one.
	if call, ok := parseCall([]byte(strings.TrimSpace(reply))); ok {
		return []Call{call}
	}
	return nil
}

func parseCall(raw []byte) (Call, bool) {
	var probe struct {
		Name       string          `json:"name"`
		Arguments  json.RawMessage `json:"arguments"`
		Parameters json.RawMessage `json:"parameters"`
	}
	if err := json.Unmarshal(raw, &probe); err != nil {
		return Call{}, false
	}
	name := strings.TrimSpace(probe.Name)
	if name == "" {
		return Call{}, false
	}
	payload := probe.Arguments
	if len(payload) == 0 {
		payload = probe.Parameters
	}
	return Call{Name: name, Arguments: decodeArguments(payload)}, true
}

func decodeArguments(raw json.RawMessage) map[string]any {
	args := map[string]any{}
	if len(raw) == 0 {
		return args
	}
	// Some models double-encode the arguments as a JSON string.
	var asString string
	if err := json.Unmarshal(raw, &asString); err == nil {
		_ = json.Unmarshal([]byte(asString), &args)
		return args
	}
	_ = json.Unmarshal(raw, &args)
	return args
}

// Agent drives one conversation.
type Agent struct {
	Client   llm.Client
	Tools    *tools.Registry
	MaxTurns int
	// Persona is appended to the system prompt when set.
	Persona string
	// Context returns extra system text, such as saved notes and open tasks.
	// It is called before every model request so it stays current.
	Context func() string
	// Out receives one line per tool call. Defaults to discarding output.
	Out io.Writer
}

const systemPrompt = `You are micro-pa, a small personal assistant running as a single program on the user's own computer.

You are direct, practical and brief. You help with everyday things: answering questions, keeping notes, tracking tasks, reading files, and running shell commands when the user has allowed it.

How to work:
1. Lead with the answer. Skip preamble, filler, and restating the question.
2. When you need information or need to change something, call a tool. Never invent a tool result, the contents of a file, or the current time.
3. Never claim something worked unless a tool told you it did. If a tool fails, read the error, then either retry with fixed arguments or say plainly what went wrong.
4. Call only the tools you need, and prefer one call at a time.
5. If a request is genuinely ambiguous, ask one short question. Otherwise make a sensible assumption and say what you assumed.
6. Save a note with the remember tool when the user tells you something durable about themselves or their work.

To call a tool, reply with a single JSON object wrapped in tool_call tags, using exactly the argument names from the schema:

<tool_call>
{"name": "recall", "arguments": {"query": "coffee"}}
</tool_call>

The result comes back in a <tool_response> message. After that, continue: either call another tool or answer the user in plain language. Never write tool_call blocks that are not real calls.

Tools available:

<tools>
%s
</tools>`

// System renders the full system prompt for this turn.
func (a *Agent) System() string {
	definitions := "[]"
	if a.Tools != nil {
		definitions = a.Tools.Definitions()
	}
	prompt := fmt.Sprintf(systemPrompt, definitions)
	if strings.TrimSpace(a.Persona) != "" {
		prompt += "\n\nAbout this user and how they want you to behave:\n" + strings.TrimSpace(a.Persona)
	}
	if a.Context != nil {
		if extra := strings.TrimSpace(a.Context()); extra != "" {
			prompt += "\n\nWhat you already know:\n" + extra
		}
	}
	return prompt
}

// Ask sends one user message and runs tool calls until the model answers. It
// returns the updated conversation; the final assistant message is the answer.
func (a *Agent) Ask(ctx context.Context, history []llm.Message, input string) ([]llm.Message, error) {
	if a.Client == nil {
		return history, fmt.Errorf("agent: no model client configured")
	}
	out := a.Out
	if out == nil {
		out = io.Discard
	}
	maxTurns := a.MaxTurns
	if maxTurns <= 0 {
		maxTurns = DefaultMaxTurns
	}

	msgs := make([]llm.Message, 0, len(history)+4)
	msgs = append(msgs, history...)
	msgs = append(msgs, llm.Message{Role: llm.RoleUser, Content: input})

	for turn := 1; turn <= maxTurns; turn++ {
		request := make([]llm.Message, 0, len(msgs)+1)
		request = append(request, llm.Message{Role: llm.RoleSystem, Content: a.System()})
		request = append(request, msgs...)

		reply, err := a.Client.Chat(ctx, request)
		if err != nil {
			return msgs, err
		}
		reply = llm.StripThinking(reply)
		msgs = append(msgs, llm.Message{Role: llm.RoleAssistant, Content: reply})

		calls := ParseToolCalls(reply)
		if len(calls) == 0 {
			return msgs, nil
		}
		if a.Tools == nil {
			msgs = append(msgs, llm.Message{
				Role:    llm.RoleTool,
				Content: "<tool_response>\nerror: no tools are available\n</tool_response>",
			})
			continue
		}

		for _, call := range calls {
			fmt.Fprintf(out, "  -> %s(%s)\n", call.Name, compactArguments(call.Arguments))
			result := a.Tools.Run(ctx, call.Name, call.Arguments)
			fmt.Fprintf(out, "  <- %s\n", oneLine(result, 200))
			msgs = append(msgs, llm.Message{Role: llm.RoleTool, Content: toolResponse(call.Name, result)})
		}
		if err := ctx.Err(); err != nil {
			return msgs, err
		}
	}

	msgs = append(msgs, llm.Message{
		Role:    llm.RoleAssistant,
		Content: fmt.Sprintf("I stopped after %d tool rounds without reaching an answer. Try narrowing the question.", maxTurns),
	})
	return msgs, nil
}

// Answer returns the text of the last assistant message.
func Answer(msgs []llm.Message) string {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == llm.RoleAssistant {
			return msgs[i].Content
		}
	}
	return ""
}

// Trim keeps the conversation small enough to send. It keeps the newest
// maxMessages entries and always starts on a user message so the model does not
// see an orphaned tool result as its first input.
func Trim(msgs []llm.Message, maxMessages int) []llm.Message {
	if maxMessages <= 0 || len(msgs) <= maxMessages {
		return msgs
	}
	trimmed := msgs[len(msgs)-maxMessages:]
	for len(trimmed) > 0 && trimmed[0].Role != llm.RoleUser {
		trimmed = trimmed[1:]
	}
	if len(trimmed) == 0 {
		return msgs[len(msgs)-1:]
	}
	return trimmed
}

func toolResponse(name, result string) string {
	payload, err := json.Marshal(struct {
		Name    string `json:"name"`
		Content string `json:"content"`
	}{Name: name, Content: result})
	if err != nil {
		return "<tool_response>\n" + result + "\n</tool_response>"
	}
	return "<tool_response>\n" + string(payload) + "\n</tool_response>"
}

func compactArguments(args map[string]any) string {
	if len(args) == 0 {
		return ""
	}
	data, err := json.Marshal(args)
	if err != nil {
		return "..."
	}
	return oneLine(string(data), 160)
}

func oneLine(s string, limit int) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) > limit {
		return s[:limit] + "..."
	}
	return s
}
