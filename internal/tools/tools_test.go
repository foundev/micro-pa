package tools

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"micro-pa/internal/store"
)

func registry(t *testing.T, o Options) *Registry {
	t.Helper()
	if o.Store == nil {
		st, err := store.Open(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		o.Store = st
	}
	reg, err := Default(o)
	if err != nil {
		t.Fatalf("Default: %v", err)
	}
	return reg
}

func TestShellIsOffByDefault(t *testing.T) {
	reg := registry(t, Options{})
	if reg.Has("shell") {
		t.Fatal("shell should not be registered unless it is enabled")
	}
	if got := reg.Run(context.Background(), "shell", map[string]any{"command": "echo hi"}); !strings.Contains(got, "unknown tool") {
		t.Errorf("running a disabled tool = %q, want an unknown-tool error", got)
	}
}

func TestShellRespectsConfirmation(t *testing.T) {
	asked := ""
	reg := registry(t, Options{
		AllowShell: true,
		Confirm: func(cmd string) bool {
			asked = cmd
			return false
		},
	})
	got := reg.Run(context.Background(), "shell", map[string]any{"command": "echo nope"})
	if asked != "echo nope" {
		t.Fatalf("confirm was asked about %q", asked)
	}
	if !strings.Contains(got, "declined") {
		t.Errorf("declined command = %q, want a declined result", got)
	}
}

func TestShellRunsWhenConfirmed(t *testing.T) {
	reg := registry(t, Options{AllowShell: true, Confirm: func(string) bool { return true }})
	got := reg.Run(context.Background(), "shell", map[string]any{"command": "echo micro-pa-ok"})

	var res struct {
		Status   string `json:"status"`
		ExitCode int    `json:"exit_code"`
		Output   string `json:"output"`
	}
	if err := json.Unmarshal([]byte(got), &res); err != nil {
		t.Fatalf("result is not JSON (%q): %v", got, err)
	}
	if res.Status != "ran" || res.ExitCode != 0 || !strings.Contains(res.Output, "micro-pa-ok") {
		t.Errorf("shell result = %+v", res)
	}
}

func TestShellReportsFailure(t *testing.T) {
	reg := registry(t, Options{AllowShell: true})
	got := reg.Run(context.Background(), "shell", map[string]any{"command": "exit 3"})
	if !strings.Contains(got, `"exit_code":3`) {
		t.Errorf("non-zero exit not reported: %q", got)
	}
	if got := reg.Run(context.Background(), "shell", map[string]any{"command": "   "}); !strings.Contains(got, "empty command") {
		t.Errorf("blank command = %q", got)
	}
}

func TestArgumentCoercion(t *testing.T) {
	st, err := store.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	reg := registry(t, Options{Store: st})
	ctx := context.Background()

	// Tags arrive as a string, a []any, or a []string depending on the model.
	cases := []any{"a,b", []any{"a", "b"}, []string{"a", "b"}}
	for _, tags := range cases {
		if got := reg.Run(ctx, "remember", map[string]any{"text": "coercion", "tags": tags}); !strings.Contains(got, "saved") {
			t.Errorf("tags %#v => %q", tags, got)
		}
	}
	// Numbers arrive as JSON floats, and as strings when a model quotes them.
	for _, limit := range []any{float64(1), "1", json.Number("1")} {
		got := reg.Run(ctx, "recall", map[string]any{"query": "coercion", "limit": limit})
		if !strings.Contains(got, `"count":1`) {
			t.Errorf("limit %#v => %q, want exactly one result", limit, got)
		}
	}
	// Booleans arrive as real booleans or as strings.
	for _, all := range []any{true, "true"} {
		if got := reg.Run(ctx, "task_list", map[string]any{"all": all}); !strings.Contains(got, `"count":0`) {
			t.Errorf("all %#v => %q", all, got)
		}
	}
}

func TestReadFileAndTruncation(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "note.txt")
	if err := os.WriteFile(path, []byte("0123456789"), 0o600); err != nil {
		t.Fatal(err)
	}
	reg := registry(t, Options{})
	ctx := context.Background()

	full := reg.Run(ctx, "read_file", map[string]any{"path": path})
	if !strings.Contains(full, "0123456789") || strings.Contains(full, "truncated") {
		t.Errorf("full read = %q", full)
	}
	partial := reg.Run(ctx, "read_file", map[string]any{"path": path, "max_bytes": float64(4)})
	if !strings.Contains(partial, "0123") || !strings.Contains(partial, `"truncated":true`) {
		t.Errorf("truncated read = %q", partial)
	}
	if got := reg.Run(ctx, "read_file", map[string]any{"path": dir}); !strings.Contains(got, "is a directory") {
		t.Errorf("directory read = %q", got)
	}
	if got := reg.Run(ctx, "read_file", map[string]any{"path": filepath.Join(dir, "missing")}); !strings.Contains(got, "error:") {
		t.Errorf("missing file = %q", got)
	}
}

func TestDefinitionsAreValidSchemas(t *testing.T) {
	reg := registry(t, Options{AllowShell: true})
	var defs []struct {
		Type     string `json:"type"`
		Function struct {
			Name        string `json:"name"`
			Description string `json:"description"`
			Parameters  struct {
				Type       string                    `json:"type"`
				Properties map[string]map[string]any `json:"properties"`
				Required   []string                  `json:"required"`
			} `json:"parameters"`
		} `json:"function"`
	}
	if err := json.Unmarshal([]byte(reg.Definitions()), &defs); err != nil {
		t.Fatalf("definitions are not valid JSON: %v", err)
	}
	if len(defs) != len(reg.Names()) {
		t.Fatalf("got %d definitions for %d tools", len(defs), len(reg.Names()))
	}
	seen := map[string]bool{}
	for _, d := range defs {
		if d.Type != "function" || d.Function.Name == "" || d.Function.Description == "" {
			t.Errorf("incomplete definition: %+v", d)
		}
		if d.Function.Parameters.Type != "object" {
			t.Errorf("%s: parameters.type = %q, want object", d.Function.Name, d.Function.Parameters.Type)
		}
		for _, req := range d.Function.Parameters.Required {
			if _, ok := d.Function.Parameters.Properties[req]; !ok {
				t.Errorf("%s: required field %q is not in properties", d.Function.Name, req)
			}
		}
		seen[d.Function.Name] = true
	}
	for _, name := range []string{"now", "remember", "recall", "forget", "task_add", "task_list", "task_done", "read_file", "shell"} {
		if !seen[name] {
			t.Errorf("tool %q is missing from the definitions", name)
		}
	}
}

func TestNowUsesInjectedClock(t *testing.T) {
	st, err := store.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	reg, err := Default(Options{Store: st})
	if err != nil {
		t.Fatal(err)
	}
	got := reg.Run(context.Background(), "now", nil)
	var res struct {
		Local   string `json:"local"`
		Weekday string `json:"weekday"`
	}
	if err := json.Unmarshal([]byte(got), &res); err != nil {
		t.Fatalf("now result = %q: %v", got, err)
	}
	if res.Local == "" || res.Weekday == "" {
		t.Errorf("now result is incomplete: %+v", res)
	}
}
