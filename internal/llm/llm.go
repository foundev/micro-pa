// Package llm talks to OpenAI-compatible chat completion endpoints and knows
// just enough about the <tool_call> calling convention to drive the agent loop.
package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"
)

// Roles used in a conversation.
const (
	RoleSystem    = "system"
	RoleUser      = "user"
	RoleAssistant = "assistant"
	RoleTool      = "tool"
)

// Message is one entry in a conversation, in OpenAI's wire format.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Client is the interface the agent depends on. It keeps the agent testable and
// lets micro-pa run fully offline with the mock client.
type Client interface {
	Chat(ctx context.Context, messages []Message) (string, error)
	Name() string
}

// OpenAIClient calls an OpenAI-compatible /chat/completions endpoint.
type OpenAIClient struct {
	BaseURL     string
	APIKey      string
	Model       string
	Temperature float64
	MaxTokens   int
	Timeout     time.Duration

	// HTTP performs the requests. When nil, a client with Timeout is used.
	// Tests set this to a transport stub so they never open a socket.
	HTTP *http.Client

	baseURL string
}

// NewOpenAIClient builds a client for one endpoint and model.
func NewOpenAIClient(baseURL, apiKey, model string, temperature float64, timeout time.Duration) *OpenAIClient {
	return &OpenAIClient{
		BaseURL:     baseURL,
		APIKey:      apiKey,
		Model:       model,
		Temperature: temperature,
		Timeout:     timeout,
		baseURL:     strings.TrimRight(baseURL, "/"),
	}
}

// Name implements Client.
func (c *OpenAIClient) Name() string { return c.Model }

// httpClient returns the client to use, defaulting to one with the configured
// timeout.
func (c *OpenAIClient) httpClient() *http.Client {
	if c.HTTP != nil {
		return c.HTTP
	}
	if c.Timeout > 0 {
		return &http.Client{Timeout: c.Timeout}
	}
	return http.DefaultClient
}

type chatRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Temperature float64   `json:"temperature,omitempty"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
}

type wireToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	} `json:"function"`
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content   string         `json:"content"`
			Reasoning string         `json:"reasoning"`
			ToolCalls []wireToolCall `json:"tool_calls"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error"`
}

// Chat sends the conversation and returns the assistant's raw reply text,
// including any <tool_call> blocks the model emitted.
func (c *OpenAIClient) Chat(ctx context.Context, messages []Message) (string, error) {
	if c.baseURL == "" {
		return "", errors.New("no base_url configured")
	}

	body, err := json.Marshal(chatRequest{
		Model:       c.Model,
		Messages:    messages,
		Temperature: c.Temperature,
		MaxTokens:   c.MaxTokens,
	})
	if err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	if c.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.APIKey)
	}
	// OpenRouter uses this for attribution in its dashboard; harmless elsewhere.
	req.Header.Set("X-Title", "micro-pa")

	resp, err := c.httpClient().Do(req)
	if err != nil {
		return "", fmt.Errorf("call %s: %w", c.baseURL, err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		return "", err
	}
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		return "", fmt.Errorf("%s: %s", resp.Status, apiErrorMessage(raw))
	}

	var parsed chatResponse
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return "", fmt.Errorf("decode response: %w", err)
	}
	if parsed.Error != nil {
		return "", fmt.Errorf("api error: %s", parsed.Error.Message)
	}
	if len(parsed.Choices) == 0 {
		return "", errors.New("api returned no choices")
	}

	msg := parsed.Choices[0].Message
	content := strings.TrimSpace(msg.Content)
	if content == "" {
		// Some providers only fill in a separate reasoning field.
		content = strings.TrimSpace(msg.Reasoning)
	}
	if native := nativeToolCalls(msg.ToolCalls); native != "" {
		if content != "" {
			content += "\n"
		}
		content += native
	}
	return content, nil
}

func apiErrorMessage(raw []byte) string {
	var parsed chatResponse
	if err := json.Unmarshal(raw, &parsed); err == nil && parsed.Error != nil {
		return parsed.Error.Message
	}
	s := strings.TrimSpace(string(raw))
	if len(s) > 300 {
		s = s[:300] + "..."
	}
	return s
}

// nativeToolCalls converts OpenAI-style tool_calls into <tool_call> blocks so
// the agent only has to understand one format.
func nativeToolCalls(calls []wireToolCall) string {
	if len(calls) == 0 {
		return ""
	}
	var sb strings.Builder
	for _, call := range calls {
		args := json.RawMessage(call.Function.Arguments)
		// Arguments sometimes arrive as a JSON string holding JSON.
		var asString string
		if err := json.Unmarshal(args, &asString); err == nil {
			args = json.RawMessage(asString)
		}
		if len(args) == 0 || !json.Valid(args) {
			args = json.RawMessage("{}")
		}
		payload, err := json.Marshal(struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		}{Name: call.Function.Name, Arguments: args})
		if err != nil {
			continue
		}
		sb.WriteString("<tool_call>\n")
		sb.Write(payload)
		sb.WriteString("\n</tool_call>\n")
	}
	return strings.TrimRight(sb.String(), "\n")
}

var thinkingBlocks = []*regexp.Regexp{
	regexp.MustCompile(`(?is)<think\b[^>]*>.*?</think\s*>`),
	regexp.MustCompile(`(?is)<thinking\b[^>]*>.*?</thinking\s*>`),
	regexp.MustCompile(`(?is)<reasoning\b[^>]*>.*?</reasoning\s*>`),
	regexp.MustCompile(`(?is)<scratchpad\b[^>]*>.*?</scratchpad\s*>`),
}

// StripThinking removes reasoning traces from a reply. Keeping them would waste
// context on every later turn, and the user does not need to read them.
func StripThinking(s string) string {
	for _, re := range thinkingBlocks {
		s = re.ReplaceAllString(s, "")
	}
	return strings.TrimSpace(s)
}

// MockClient is a tiny offline stand-in for a real model. It lets you try the
// whole loop (including tools) with no network and no API key.
type MockClient struct{}

// Name implements Client.
func (MockClient) Name() string { return "mock" }

// Chat implements Client with keyword matching, which is enough to exercise
// every tool at least once.
func (MockClient) Chat(_ context.Context, messages []Message) (string, error) {
	if len(messages) == 0 {
		return "", errors.New("no messages")
	}
	last := messages[len(messages)-1]
	if last.Role == RoleTool {
		return "Done - the tool result above is what you asked for.", nil
	}

	q := strings.ToLower(last.Content)
	switch {
	case strings.Contains(q, "time") || strings.Contains(q, "date"):
		return "Let me check the clock.\n" + callBlock("now", map[string]any{}), nil
	case strings.Contains(q, "remember"):
		return "Saving that.\n" + callBlock("remember", map[string]any{
			"text": strings.TrimSpace(last.Content),
			"tags": []string{"chat"},
		}), nil
	case strings.Contains(q, "recall") || strings.Contains(q, "what do you know"):
		return "Searching my notes.\n" + callBlock("recall", map[string]any{"query": last.Content}), nil
	case strings.Contains(q, "task") || strings.Contains(q, "todo") || strings.Contains(q, "remind"):
		return "Adding a task.\n" + callBlock("task_add", map[string]any{
			"text": strings.TrimSpace(last.Content),
		}), nil
	case strings.Contains(q, "list") && strings.Contains(q, "file"):
		return "Listing files.\n" + callBlock("shell", map[string]any{"command": "ls -la"}), nil
	default:
		return "I am the mock provider, so I cannot really think.\n\n" +
			"Try: \"what time is it\", \"remember that I like short answers\", " +
			"\"what do you know about me\", \"task: buy milk\", or \"list the files here\".\n\n" +
			"For a real conversation set a provider and key, e.g. " +
			"MPA_PROVIDER=openai MPA_API_KEY=sk-or-... mpa.", nil
	}
}

func callBlock(name string, args map[string]any) string {
	payload, err := json.Marshal(struct {
		Name      string         `json:"name"`
		Arguments map[string]any `json:"arguments"`
	}{Name: name, Arguments: args})
	if err != nil {
		return ""
	}
	return "<tool_call>\n" + string(payload) + "\n</tool_call>"
}
