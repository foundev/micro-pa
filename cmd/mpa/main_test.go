package main

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"micro-pa/internal/config"
	"micro-pa/internal/llm"
)

// isolate clears every environment variable the CLI reads and points the state
// directory at a temp dir.
func isolate(t *testing.T) string {
	t.Helper()
	home := t.TempDir()
	for _, key := range []string{
		"MPA_PROVIDER", "MPA_BASE_URL", "MPA_MODEL", "MPA_PERSONA", "MPA_API_KEY",
		"MPA_ALLOW_SHELL", "MPA_MAX_TURNS", "MPA_TIMEOUT_SECONDS", "MPA_TEMPERATURE",
		"OPENROUTER_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL", "MPA_HOME",
	} {
		t.Setenv(key, "")
	}
	t.Setenv("MPA_HOME", home)
	return home
}

// runCLI runs the command and returns its exit code with captured output.
func runCLI(t *testing.T, stdin string, args ...string) (int, string, string) {
	t.Helper()
	var out, errOut bytes.Buffer
	code := run(args, strings.NewReader(stdin), &out, &errOut)
	return code, out.String(), errOut.String()
}

// record is what the stub endpoint saw on one request.
type record struct {
	Path     string
	Auth     string
	Model    string
	Messages []llm.Message
}

// stubEndpoint captures the requests the real client makes and replies with a
// scripted sequence. It runs inside the process as an http.RoundTripper, so the
// whole request-building path is exercised without opening a socket.
type stubEndpoint struct {
	mu      sync.Mutex
	replies []string
	// script, when set, decides the reply from the incoming conversation. It
	// is how a test makes the stand-in model call a tool.
	script func([]llm.Message) string
	seen   []record
	idx    int
}

func (s *stubEndpoint) RoundTrip(r *http.Request) (*http.Response, error) {
	body, _ := io.ReadAll(r.Body)
	var req struct {
		Model    string        `json:"model"`
		Messages []llm.Message `json:"messages"`
	}
	_ = json.Unmarshal(body, &req)

	s.mu.Lock()
	s.seen = append(s.seen, record{
		Path:     r.URL.Path,
		Auth:     r.Header.Get("Authorization"),
		Model:    req.Model,
		Messages: req.Messages,
	})
	reply := ""
	switch {
	case s.script != nil:
		reply = s.script(req.Messages)
	case s.idx < len(s.replies):
		reply = s.replies[s.idx]
	}
	s.idx++
	s.mu.Unlock()

	payload, _ := json.Marshal(map[string]any{
		"choices": []map[string]any{{
			"message":       map[string]any{"content": reply},
			"finish_reason": "stop",
		}},
	})
	return &http.Response{
		StatusCode: 200,
		Status:     "200 OK",
		Body:       io.NopCloser(bytes.NewReader(payload)),
		Header:     http.Header{"Content-Type": []string{"application/json"}},
	}, nil
}

// useStub points the CLI at the stub endpoint for the duration of the test.
func useStub(t *testing.T, replies ...string) *stubEndpoint {
	t.Helper()
	return useStubWith(t, nil, replies)
}

// useStubWith is useStub with an explicit reply strategy.
func useStubWith(t *testing.T, script func([]llm.Message) string, replies []string) *stubEndpoint {
	t.Helper()
	stub := &stubEndpoint{replies: replies, script: script}
	previous := clientBuilder
	clientBuilder = func(cfg *config.Config) llm.Client {
		client := llm.NewOpenAIClient(cfg.BaseURL, cfg.APIKey, cfg.Model, cfg.Temperature, 0)
		client.HTTP = &http.Client{Transport: stub}
		return client
	}
	t.Cleanup(func() { clientBuilder = previous })
	return stub
}

func TestOnceAgainstStubServer(t *testing.T) {
	home := isolate(t)
	stub := useStub(t,
		`<tool_call>{"name":"task_add","arguments":{"text":"renew the domain"}}</tool_call>`,
		"Added that to your list.",
	)

	code, out, errOut := runCLI(t, "", "-once", "add a task to renew the domain",
		"-provider", "openai",
		"-base-url", "https://stub.invalid/v1",
		"-api-key", "test-key",
		"-model", "deepseek/deepseek-v4-flash",
	)
	if code != 0 {
		t.Fatalf("exit code = %d, stderr:\n%s", code, errOut)
	}
	if !strings.Contains(out, "Added that to your list.") {
		t.Errorf("stdout did not contain the answer:\n%s", out)
	}
	if !strings.Contains(errOut, "task_add") {
		t.Errorf("the tool call was not traced to stderr:\n%s", errOut)
	}

	requests := stub.seen
	if len(requests) != 2 {
		t.Fatalf("expected two model requests, got %d", len(requests))
	}
	if got := requests[0].Path; got != "/v1/chat/completions" {
		t.Errorf("path = %v", got)
	}
	if got := requests[0].Auth; got != "Bearer test-key" {
		t.Errorf("auth header = %v", got)
	}
	if got := requests[0].Model; got != "deepseek/deepseek-v4-flash" {
		t.Errorf("model = %v", got)
	}
	// The second request must carry the tool result back to the model.
	msgs := requests[1].Messages
	if len(msgs) < 2 {
		t.Fatalf("second request has %d messages", len(msgs))
	}
	last := msgs[len(msgs)-1]
	if last.Role != "tool" || !strings.Contains(last.Content, "<tool_response>") {
		t.Errorf("last message = %+v, want a tool response", last)
	}

	// State and session are written on disk.
	data, err := os.ReadFile(filepath.Join(home, "tasks.json"))
	if err != nil {
		t.Fatalf("tasks.json: %v", err)
	}
	if !strings.Contains(string(data), "renew the domain") {
		t.Errorf("task was not persisted:\n%s", data)
	}
	if _, err := os.Stat(filepath.Join(home, "sessions", "default.json")); err != nil {
		t.Errorf("session was not saved: %v", err)
	}
}

func TestMemoryCarriesIntoTheSystemPrompt(t *testing.T) {
	isolate(t)
	// A stand-in model that calls the remember tool when asked to, and
	// otherwise just answers.
	stub := useStubWith(t, func(msgs []llm.Message) string {
		last := msgs[len(msgs)-1]
		if last.Role == llm.RoleTool {
			return "Noted."
		}
		if strings.Contains(strings.ToLower(last.Content), "remember") {
			return `<tool_call>{"name":"remember","arguments":{"text":"prefers dark mode"}}</tool_call>`
		}
		return "ok"
	}, nil)

	code, _, errOut := runCLI(t, "", "-provider", "openai", "-base-url", "https://stub.invalid/v1",
		"-api-key", "k", "-model", "m", "-once", "remember that I prefer dark mode")
	if code != 0 {
		t.Fatalf("memory run failed: %s", errOut)
	}

	code, _, errOut = runCLI(t, "", "-provider", "openai", "-base-url", "https://stub.invalid/v1",
		"-api-key", "k", "-model", "m", "-once", "what do I like")
	if code != 0 {
		t.Fatalf("final run failed: %s", errOut)
	}

	requests := stub.seen
	last := requests[len(requests)-1]
	system := last.Messages[0]
	if system.Role != "system" {
		t.Fatalf("first message is not the system prompt: %+v", system)
	}
	if !strings.Contains(system.Content, "prefers dark mode") {
		t.Errorf("saved note missing from the system prompt:\n%s", system.Content)
	}
}

func TestMissingKeyExplainsItself(t *testing.T) {
	isolate(t)
	code, out, errOut := runCLI(t, "", "-once", "hi")
	if code != 1 {
		t.Errorf("exit code = %d, want 1", code)
	}
	if !strings.Contains(errOut, "no API key") || !strings.Contains(errOut, "OPENROUTER_API_KEY") {
		t.Errorf("stderr should explain how to set a key:\n%s", errOut)
	}
	if strings.Contains(out, "error") {
		t.Errorf("the error should go to stderr, not stdout:\n%s", out)
	}
}

func TestMockProviderRunsOffline(t *testing.T) {
	isolate(t)
	code, out, errOut := runCLI(t, "", "-provider", "mock", "-once", "what time is it")
	if code != 0 {
		t.Fatalf("exit code = %d: %s", code, errOut)
	}
	if !strings.Contains(out, "Done -") {
		t.Errorf("mock reply missing:\n%s", out)
	}
	if !strings.Contains(errOut, "-> now()") {
		t.Errorf("the mock should have called the now tool:\n%s", errOut)
	}
}

func TestToolAndSessionListing(t *testing.T) {
	isolate(t)
	code, out, _ := runCLI(t, "", "-provider", "mock", "-tools")
	if code != 0 {
		t.Fatalf("exit code = %d", code)
	}
	for _, want := range []string{"now", "remember", "task_add", "read_file"} {
		if !strings.Contains(out, want) {
			t.Errorf("-tools output is missing %q:\n%s", want, out)
		}
	}
	for _, line := range strings.Split(out, "\n") {
		if strings.TrimSpace(line) == "shell" {
			t.Errorf("shell should be hidden until it is enabled:\n%s", out)
		}
	}

	code, out, _ = runCLI(t, "", "-provider", "mock", "-sessions")
	if code != 0 || !strings.Contains(out, "no saved conversations") {
		t.Errorf("empty -sessions output = %q (code %d)", out, code)
	}
}

func TestREPLSlashCommands(t *testing.T) {
	home := isolate(t)
	code, out, errOut := runCLI(t, "/remember tabs over spaces\n/memory tabs\n/quit\n", "-provider", "mock")
	if code != 0 {
		t.Fatalf("exit code = %d: %s", code, errOut)
	}
	if !strings.Contains(out, "saved m_") {
		t.Errorf("/remember did not confirm:\n%s", out)
	}
	if !strings.Contains(out, "tabs over spaces") {
		t.Errorf("/memory did not list the note:\n%s", out)
	}
	data, err := os.ReadFile(filepath.Join(home, "memory.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "tabs over spaces") {
		t.Errorf("note was not persisted:\n%s", data)
	}
}

func TestPositionalPromptStartsTheSession(t *testing.T) {
	isolate(t)
	code, out, errOut := runCLI(t, "/quit\n", "-provider", "mock", "what time is it")
	if code != 0 {
		t.Fatalf("exit code = %d: %s", code, errOut)
	}
	if !strings.Contains(errOut, "-> now()") {
		t.Errorf("the positional prompt was not sent to the model:\n%s", errOut)
	}
	if !strings.Contains(out, "micro-pa") {
		t.Errorf("the banner is missing:\n%s", out)
	}
}

func TestInitWritesConfigOnce(t *testing.T) {
	home := isolate(t)
	if code, _, errOut := runCLI(t, "", "-init"); code != 0 {
		t.Fatalf("first -init failed: %s", errOut)
	}
	if _, err := os.Stat(filepath.Join(home, "config.json")); err != nil {
		t.Fatalf("config was not written: %v", err)
	}
	code, _, errOut := runCLI(t, "", "-init")
	if code != 1 || !strings.Contains(errOut, "already exists") {
		t.Errorf("second -init: code=%d stderr=%s", code, errOut)
	}
}

func TestUsageAndVersion(t *testing.T) {
	isolate(t)
	if code, out, _ := runCLI(t, "", "-version"); code != 0 || !strings.Contains(out, version) {
		t.Errorf("-version: code=%d out=%q", code, out)
	}
	if code, _, errOut := runCLI(t, "", "-h"); code != 0 || !strings.Contains(errOut, "Usage:") {
		t.Errorf("-h: code=%d stderr=%q", code, errOut)
	}
	if code, _, _ := runCLI(t, "", "-nope"); code != 2 {
		t.Errorf("unknown flag should exit 2, got %d", code)
	}
}

func TestShellRequiresApprovalInTheREPL(t *testing.T) {
	isolate(t)
	stdin := "/shell on\nlist the files\ny\n/shell off\n/quit\n"
	code, out, errOut := runCLI(t, stdin, "-provider", "mock")
	if code != 0 {
		t.Fatalf("exit code = %d: %s", code, errOut)
	}
	if !strings.Contains(out, "run this command?") {
		t.Errorf("the user was not asked to approve the command:\n%s", out)
	}
	if !strings.Contains(errOut, "exit_code") {
		t.Errorf("approved command did not run:\n%s", errOut)
	}

	// Saying no must skip the command.
	code, out, errOut = runCLI(t, "/shell on\nlist the files\nn\n/quit\n", "-provider", "mock")
	if code != 0 {
		t.Fatalf("exit code = %d: %s", code, errOut)
	}
	if !strings.Contains(out, "declined") {
		t.Errorf("declining was not reported:\n%s", out)
	}
	if strings.Contains(errOut, `"status":"ran"`) {
		t.Errorf("a declined command still ran:\n%s", errOut)
	}
}
