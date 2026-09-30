package llm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
)

// roundTripFunc lets a test stand in for the network.
type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func respond(t *testing.T, status int, body string) *OpenAIClient {
	t.Helper()
	c := NewOpenAIClient("https://example.invalid/v1", "key", "model", 0.5, 0)
	c.HTTP = &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: status,
			Status:     http.StatusText(status),
			Body:       io.NopCloser(strings.NewReader(body)),
			Header:     http.Header{},
		}, nil
	})}
	return c
}

func TestChatPlainContent(t *testing.T) {
	c := respond(t, 200, `{"choices":[{"message":{"content":"  hello  "},"finish_reason":"stop"}]}`)
	got, err := c.Chat(context.Background(), []Message{{Role: RoleUser, Content: "hi"}})
	if err != nil {
		t.Fatal(err)
	}
	if got != "hello" {
		t.Errorf("content = %q, want it trimmed", got)
	}
}

func TestChatFallsBackToReasoning(t *testing.T) {
	c := respond(t, 200, `{"choices":[{"message":{"content":"","reasoning":"thought it through"},"finish_reason":"stop"}]}`)
	got, err := c.Chat(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if got != "thought it through" {
		t.Errorf("reasoning fallback = %q", got)
	}
}

func TestNativeToolCallsBecomeBlocks(t *testing.T) {
	body := `{"choices":[{"message":{"content":"Calling a tool.",
      "tool_calls":[{"id":"1","function":{"name":"now","arguments":"{}"}},
                    {"id":"2","function":{"name":"recall","arguments":"{\"query\":\"coffee\"}"}}]},
      "finish_reason":"tool_calls"}]}`
	c := respond(t, 200, body)
	got, err := c.Chat(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got, "Calling a tool.") {
		t.Errorf("prose was dropped: %q", got)
	}
	calls := ParseToolCallsForTest(got)
	if len(calls) != 2 || calls[0]["name"] != "now" {
		t.Fatalf("converted calls = %+v", calls)
	}
	// Arguments that arrive as a nested JSON string must be decoded.
	args, _ := calls[1]["arguments"].(map[string]any)
	if args["query"] != "coffee" {
		t.Errorf("nested arguments were not decoded: %+v", args)
	}
}

// ParseToolCallsForTest decodes each <tool_call> block, which is how the agent
// package reads them. It avoids importing the agent package from here.
func ParseToolCallsForTest(reply string) []map[string]any {
	var out []map[string]any
	for _, chunk := range strings.Split(reply, "<tool_call>")[1:] {
		chunk = strings.SplitN(chunk, "</tool_call>", 2)[0]
		var parsed map[string]any
		if err := json.Unmarshal([]byte(strings.TrimSpace(chunk)), &parsed); err == nil {
			out = append(out, parsed)
		}
	}
	return out
}

func TestChatHTTPErrorIncludesAPIMessage(t *testing.T) {
	c := respond(t, 401, `{"error":{"message":"invalid api key"}}`)
	_, err := c.Chat(context.Background(), nil)
	if err == nil || !strings.Contains(err.Error(), "invalid api key") {
		t.Errorf("error = %v, want the API message", err)
	}
}

func TestChatRejectsEmptyAndMalformedBodies(t *testing.T) {
	if _, err := respond(t, 200, `{"choices":[]}`).Chat(context.Background(), nil); err == nil {
		t.Error("expected an error when there are no choices")
	}
	if _, err := respond(t, 200, `not json`).Chat(context.Background(), nil); err == nil {
		t.Error("expected a decode error")
	}
	c := NewOpenAIClient("", "k", "m", 0, 0)
	if _, err := c.Chat(context.Background(), nil); err == nil {
		t.Error("expected an error when no base URL is set")
	}
}

func TestChatSendsAuthAndPayload(t *testing.T) {
	var got map[string]any
	var auth string
	c := NewOpenAIClient("https://example.invalid/v1", "sk-test", "deepseek/deepseek-v4-flash", 0.3, 0)
	c.HTTP = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		auth = r.Header.Get("Authorization")
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &got)
		return &http.Response{
			StatusCode: 200,
			Status:     "200 OK",
			Body:       io.NopCloser(strings.NewReader(`{"choices":[{"message":{"content":"ok"}}]}`)),
			Header:     http.Header{},
		}, nil
	})}
	if _, err := c.Chat(context.Background(), []Message{{Role: RoleSystem, Content: "s"}}); err != nil {
		t.Fatal(err)
	}
	if auth != "Bearer sk-test" {
		t.Errorf("authorization = %q", auth)
	}
	if got["model"] != "deepseek/deepseek-v4-flash" || got["temperature"] != 0.3 {
		t.Errorf("payload = %+v", got)
	}
}

func TestStripThinking(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{
			name: "think block removed",
			in:   "Before <think>hidden reasoning</think> after",
			want: "Before  after",
		},
		{
			name: "multiline reasoning removed",
			in:   "Answer.\n<reasoning>\nstep 1\nstep 2\n</reasoning>",
			want: "Answer.",
		},
		{
			name: "tool calls survive",
			in:   "<thinking>hmm</thinking><tool_call>{\"name\":\"now\"}</tool_call>",
			want: `<tool_call>{"name":"now"}</tool_call>`,
		},
		{
			name: "plain text untouched",
			in:   "just an answer",
			want: "just an answer",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := StripThinking(tc.in); got != tc.want {
				t.Errorf("StripThinking = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestMockClientHitsEachTool(t *testing.T) {
	ctx := context.Background()
	tests := []struct {
		prompt string
		tool   string
	}{
		{"what time is it", "now"},
		{"remember that I like tea", "remember"},
		{"recall my notes about tea", "recall"},
		{"add a task to buy milk", "task_add"},
	}
	for _, tc := range tests {
		reply, err := MockClient{}.Chat(ctx, []Message{{Role: RoleUser, Content: tc.prompt}})
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(reply, `"`+tc.tool+`"`) {
			t.Errorf("prompt %q did not call %s:\n%s", tc.prompt, tc.tool, reply)
		}
	}
	// After a tool result, the mock stops calling tools.
	reply, _ := MockClient{}.Chat(ctx, []Message{{Role: RoleTool, Content: "x"}})
	if strings.Contains(reply, "<tool_call>") {
		t.Errorf("the mock should not loop: %q", reply)
	}
	// Empty input is an error, not a panic.
	if _, err := (MockClient{}).Chat(ctx, nil); err == nil {
		t.Error("expected an error for an empty conversation")
	}
}
