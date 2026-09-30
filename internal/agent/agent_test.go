package agent

import (
	"strings"
	"testing"

	"micro-pa/internal/llm"
)

func TestParseToolCalls(t *testing.T) {
	tests := []struct {
		name     string
		reply    string
		wantLen  int
		wantName string
		wantArg  string
	}{
		{
			name:     "tag form",
			reply:    "<tool_call>{\"name\":\"now\",\"arguments\":{}}</tool_call>",
			wantLen:  1,
			wantName: "now",
		},
		{
			name:     "tag form with surrounding prose",
			reply:    "Let me check.\n<tool_call>\n{\"name\": \"recall\", \"arguments\": {\"query\": \"coffee\"}}\n</tool_call>\nThere.",
			wantLen:  1,
			wantName: "recall",
			wantArg:  "coffee",
		},
		{
			name:     "fenced json fallback",
			reply:    "Sure:\n```json\n{\"name\": \"task_list\", \"arguments\": {\"all\": true}}\n```",
			wantLen:  1,
			wantName: "task_list",
			wantArg:  "true",
		},
		{
			name:     "bare json object",
			reply:    `{"name": "now", "arguments": {}}`,
			wantLen:  1,
			wantName: "now",
		},
		{
			name:     "double encoded arguments",
			reply:    `<tool_call>{"name":"recall","arguments":"{\"query\":\"needle\"}"}</tool_call>`,
			wantLen:  1,
			wantName: "recall",
			wantArg:  "needle",
		},
		{
			name:    "two calls",
			reply:   "<tool_call>{\"name\":\"now\",\"arguments\":{}}</tool_call><tool_call>{\"name\":\"task_list\",\"arguments\":{}}</tool_call>",
			wantLen: 2,
		},
		{
			name:    "plain prose is not a call",
			reply:   "I have no idea what you mean, and {} is not a call.",
			wantLen: 0,
		},
		{
			name:    "malformed json is ignored",
			reply:   "<tool_call>{not json}</tool_call>",
			wantLen: 0,
		},
		{
			name:    "missing name is ignored",
			reply:   "<tool_call>{\"arguments\":{\"a\":1}}</tool_call>",
			wantLen: 0,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := ParseToolCalls(tc.reply)
			if len(got) != tc.wantLen {
				t.Fatalf("got %d calls (%+v), want %d", len(got), got, tc.wantLen)
			}
			if tc.wantName != "" && got[0].Name != tc.wantName {
				t.Errorf("name = %q, want %q", got[0].Name, tc.wantName)
			}
			if tc.wantArg != "" {
				if v, ok := got[0].Arguments["query"]; ok {
					if v != tc.wantArg {
						t.Errorf("query = %v, want %q", v, tc.wantArg)
					}
				} else if v, ok := got[0].Arguments["all"]; !ok || v != true {
					t.Errorf("all = %v, want true", v)
				}
			}
		})
	}
}

func TestTrimStartsOnUserMessage(t *testing.T) {
	history := []llm.Message{
		{Role: llm.RoleUser, Content: "a"},
		{Role: llm.RoleAssistant, Content: "b"},
		{Role: llm.RoleTool, Content: "c"},
		{Role: llm.RoleUser, Content: "d"},
		{Role: llm.RoleAssistant, Content: "e"},
	}
	got := Trim(history, 3)
	if len(got) != 2 || got[0].Content != "d" {
		t.Fatalf("Trim = %+v, want the last two messages starting at \"d\"", got)
	}
	if short := Trim(history, 10); len(short) != len(history) {
		t.Errorf("Trim should not shorten a short history")
	}
}

func TestAnswer(t *testing.T) {
	msgs := []llm.Message{
		{Role: llm.RoleUser, Content: "q"},
		{Role: llm.RoleTool, Content: "t"},
		{Role: llm.RoleAssistant, Content: "the answer"},
	}
	if got := Answer(msgs); got != "the answer" {
		t.Errorf("Answer = %q", got)
	}
	if got := Answer([]llm.Message{{Role: llm.RoleUser, Content: "q"}}); got != "" {
		t.Errorf("Answer with no assistant message = %q, want empty", got)
	}
}

func TestSystemPromptIncludesToolsAndPersona(t *testing.T) {
	a := &Agent{
		Persona: "I prefer metric units.",
		Context: func() string { return "- [m_1] likes tabs" },
	}
	prompt := a.System()
	for _, want := range []string{"micro-pa", "I prefer metric units.", "[m_1] likes tabs", "<tool_call>"} {
		if !strings.Contains(prompt, want) {
			t.Errorf("system prompt missing %q", want)
		}
	}
}

func TestAskWithoutClientFails(t *testing.T) {
	a := &Agent{}
	if _, err := a.Ask(t.Context(), nil, "hi"); err == nil {
		t.Error("expected an error when no client is configured")
	}
}
