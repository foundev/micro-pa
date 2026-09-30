// Package tools implements the capabilities micro-pa hands to the model.
//
// Each tool is a plain Go function plus a small JSON schema. The registry can
// render those schemas for the system prompt and run a call by name. Tools never
// return an error to the agent: a failure is turned into a short text result so
// the model can read it and recover.
package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sort"
	"strconv"
	"strings"
	"time"

	"micro-pa/internal/store"
)

const (
	// defaultReadLimit is how much of a file read_file returns by default.
	defaultReadLimit = 16 << 10
	// maxReadLimit caps how much of a file a single call can return.
	maxReadLimit = 256 << 10
	// defaultShellTimeout and maxShellTimeout bound a single shell command.
	defaultShellTimeout = 20 * time.Second
	maxShellTimeout     = 120 * time.Second
	// maxShellOutput caps captured command output kept for the model.
	maxShellOutput = 8 << 10
)

// Tool is one callable capability.
type Tool struct {
	Name        string
	Description string
	Properties  map[string]any
	Required    []string
	Run         func(ctx context.Context, args map[string]any) (string, error)
}

// Registry holds every tool micro-pa can call.
type Registry struct {
	tools   map[string]Tool
	order   []string
	confirm func(prompt string) bool
}

// Options controls which tools are available.
type Options struct {
	Store *store.Store
	// AllowShell turns on the shell tool. It is off unless asked for.
	AllowShell bool
	// Confirm is asked before running a shell command. When nil, commands run
	// without confirmation (useful for one-shot, non-interactive runs).
	Confirm func(prompt string) bool
	// Now lets tests pin the clock.
	Now func() time.Time
}

// Default builds the standard tool set.
func Default(o Options) (*Registry, error) {
	if o.Store == nil {
		return nil, fmt.Errorf("tools: store is required")
	}
	if o.Now == nil {
		o.Now = time.Now
	}
	r := &Registry{tools: map[string]Tool{}, confirm: o.Confirm}

	r.add(Tool{
		Name:        "now",
		Description: "Current local date and time, plus the weekday and timezone. Use it whenever the answer depends on the current time.",
		Properties:  map[string]any{},
		Run: func(_ context.Context, _ map[string]any) (string, error) {
			n := o.Now()
			zone, _ := n.Zone()
			return jsonResult(map[string]any{
				"local":    n.Format(time.RFC3339),
				"weekday":  n.Weekday().String(),
				"utc":      n.UTC().Format(time.RFC3339),
				"unix":     n.Unix(),
				"timezone": zone,
			})
		},
	})

	r.add(Tool{
		Name:        "remember",
		Description: "Save a durable note about the user or their work, so it is available in later conversations.",
		Properties: map[string]any{
			"text": schema("string", "The fact or note to remember, written as a standalone sentence."),
			"tags": map[string]any{
				"type":        "array",
				"items":       map[string]any{"type": "string"},
				"description": "Optional short tags, for example [\"work\", \"preference\"].",
			},
		},
		Required: []string{"text"},
		Run: func(_ context.Context, args map[string]any) (string, error) {
			m, err := o.Store.AddMemory(argString(args, "text"), argStrings(args, "tags"))
			if err != nil {
				return "", err
			}
			return jsonResult(map[string]any{"status": "saved", "id": m.ID, "text": m.Text})
		},
	})

	r.add(Tool{
		Name:        "recall",
		Description: "Search saved notes. Call this when the user refers to something you might have been told before, or asks what you know.",
		Properties: map[string]any{
			"query": schema("string", "Words to search for. Leave empty to get the most recent notes."),
			"limit": schema("integer", "Maximum number of notes to return. Defaults to 8."),
		},
		Required: []string{"query"},
		Run: func(_ context.Context, args map[string]any) (string, error) {
			found, err := o.Store.SearchMemories(argString(args, "query"), argInt(args, "limit", 0))
			if err != nil {
				return "", err
			}
			return jsonResult(map[string]any{"count": len(found), "memories": found})
		},
	})

	r.add(Tool{
		Name:        "forget",
		Description: "Delete a saved note by its id. Use this when the user asks you to forget something.",
		Properties: map[string]any{
			"id": schema("string", "Id of the note, as returned by recall."),
		},
		Required: []string{"id"},
		Run: func(_ context.Context, args map[string]any) (string, error) {
			m, ok, err := o.Store.ForgetMemory(argString(args, "id"))
			if err != nil {
				return "", err
			}
			if !ok {
				return jsonResult(map[string]any{"status": "not_found"})
			}
			return jsonResult(map[string]any{"status": "forgotten", "id": m.ID, "text": m.Text})
		},
	})

	r.add(Tool{
		Name:        "task_add",
		Description: "Add a task to the user's todo list.",
		Properties: map[string]any{
			"text": schema("string", "What needs to be done."),
			"due":  schema("string", "Optional due date, as free text such as 2026-10-04 or next friday."),
		},
		Required: []string{"text"},
		Run: func(_ context.Context, args map[string]any) (string, error) {
			t, err := o.Store.AddTask(argString(args, "text"), argString(args, "due"))
			if err != nil {
				return "", err
			}
			return jsonResult(map[string]any{"status": "added", "id": t.ID, "text": t.Text, "due": t.Due})
		},
	})

	r.add(Tool{
		Name:        "task_list",
		Description: "List tasks. Open tasks come first. Use this before closing a task so you have the right id.",
		Properties: map[string]any{
			"all": schema("boolean", "Set true to include completed tasks. Defaults to open tasks only."),
		},
		Run: func(_ context.Context, args map[string]any) (string, error) {
			all, err := o.Store.Tasks()
			if err != nil {
				return "", err
			}
			includeDone := argBool(args, "all", false)
			out := make([]store.Task, 0, len(all))
			for _, t := range all {
				if t.Done && !includeDone {
					continue
				}
				out = append(out, t)
			}
			return jsonResult(map[string]any{"count": len(out), "tasks": out})
		},
	})

	r.add(Tool{
		Name:        "task_done",
		Description: "Mark a task done, or reopen it with done=false.",
		Properties: map[string]any{
			"id":   schema("string", "Id of the task, as returned by task_list or task_add."),
			"done": schema("boolean", "True to complete the task, false to reopen it. Defaults to true."),
		},
		Required: []string{"id"},
		Run: func(_ context.Context, args map[string]any) (string, error) {
			t, ok, err := o.Store.SetTaskDone(argString(args, "id"), argBool(args, "done", true))
			if err != nil {
				return "", err
			}
			if !ok {
				return jsonResult(map[string]any{"status": "not_found"})
			}
			return jsonResult(map[string]any{"status": "updated", "id": t.ID, "text": t.Text, "done": t.Done})
		},
	})

	r.add(Tool{
		Name:        "read_file",
		Description: "Read a text file and return its contents. Use this instead of guessing what a file says.",
		Properties: map[string]any{
			"path":      schema("string", "Path to the file, absolute or relative to the working directory."),
			"max_bytes": schema("integer", "Maximum bytes to return. Defaults to 16384, capped at 262144."),
		},
		Required: []string{"path"},
		Run: func(_ context.Context, args map[string]any) (string, error) {
			return readFile(argString(args, "path"), argInt(args, "max_bytes", defaultReadLimit))
		},
	})

	if o.AllowShell {
		r.add(Tool{
			Name:        "shell",
			Description: "Run a shell command with sh -c and return its output. Requires the user to have enabled shell access.",
			Properties: map[string]any{
				"command":         schema("string", "The command line to run."),
				"timeout_seconds": schema("integer", "Seconds before the command is killed. Defaults to 20, capped at 120."),
			},
			Required: []string{"command"},
			Run: func(ctx context.Context, args map[string]any) (string, error) {
				return r.runShell(ctx, argString(args, "command"), argInt(args, "timeout_seconds", int(defaultShellTimeout/time.Second)))
			},
		})
	}

	sort.Strings(r.order)
	return r, nil
}

func (r *Registry) add(t Tool) {
	r.tools[t.Name] = t
	r.order = append(r.order, t.Name)
}

// Names returns every registered tool name, sorted.
func (r *Registry) Names() []string {
	out := make([]string, len(r.order))
	copy(out, r.order)
	return out
}

// Has reports whether a tool is registered.
func (r *Registry) Has(name string) bool {
	_, ok := r.tools[name]
	return ok
}

// Definitions renders every tool as an OpenAI-style function schema. It is
// embedded in the system prompt so the model knows what it can call.
func (r *Registry) Definitions() string {
	type definition struct {
		Type     string      `json:"type"`
		Function functionDef `json:"function"`
	}
	defs := make([]definition, 0, len(r.order))
	for _, name := range r.order {
		t := r.tools[name]
		props := t.Properties
		if props == nil {
			props = map[string]any{}
		}
		defs = append(defs, definition{
			Type: "function",
			Function: functionDef{
				Name:        t.Name,
				Description: t.Description,
				Parameters: functionSchema{
					Type:       "object",
					Properties: props,
					Required:   t.Required,
				},
			},
		})
	}
	data, err := json.MarshalIndent(defs, "", "  ")
	if err != nil {
		return "[]"
	}
	return string(data)
}

type functionDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  functionSchema `json:"parameters"`
}

type functionSchema struct {
	Type       string         `json:"type"`
	Properties map[string]any `json:"properties"`
	Required   []string       `json:"required,omitempty"`
}

// Run executes a tool by name and returns a result string. Unknown tools and
// errors become readable text rather than Go errors, so the model can react.
func (r *Registry) Run(ctx context.Context, name string, args map[string]any) string {
	t, ok := r.tools[name]
	if !ok {
		return fmt.Sprintf("error: unknown tool %q", name)
	}
	if args == nil {
		args = map[string]any{}
	}
	out, err := t.Run(ctx, args)
	if err != nil {
		return "error: " + err.Error()
	}
	return out
}

func (r *Registry) runShell(ctx context.Context, command string, timeoutSec int) (string, error) {
	command = strings.TrimSpace(command)
	if command == "" {
		return "", fmt.Errorf("empty command")
	}
	if r.confirm != nil && !r.confirm(command) {
		return jsonResult(map[string]any{"status": "declined", "reason": "the user declined to run this command"})
	}
	if timeoutSec <= 0 {
		timeoutSec = int(defaultShellTimeout / time.Second)
	}
	if timeoutSec > int(maxShellTimeout/time.Second) {
		timeoutSec = int(maxShellTimeout / time.Second)
	}

	runCtx, cancel := context.WithTimeout(ctx, time.Duration(timeoutSec)*time.Second)
	defer cancel()

	cmd := exec.CommandContext(runCtx, "sh", "-c", command)
	raw, err := cmd.CombinedOutput()
	output := string(raw)
	truncated := false
	if len(output) > maxShellOutput {
		output = output[:maxShellOutput]
		truncated = true
	}

	result := map[string]any{
		"command":   command,
		"exit_code": cmd.ProcessState.ExitCode(),
		"output":    output,
	}
	if truncated {
		result["truncated"] = true
	}
	switch {
	case runCtx.Err() == context.DeadlineExceeded:
		result["status"] = "timeout"
		result["note"] = fmt.Sprintf("killed after %ds", timeoutSec)
	case err != nil && cmd.ProcessState == nil:
		result["status"] = "failed_to_start"
		result["note"] = err.Error()
	default:
		result["status"] = "ran"
	}
	return jsonResult(result)
}

func readFile(path string, maxBytes int) (string, error) {
	if strings.TrimSpace(path) == "" {
		return "", fmt.Errorf("empty path")
	}
	if maxBytes <= 0 {
		maxBytes = defaultReadLimit
	}
	if maxBytes > maxReadLimit {
		maxBytes = maxReadLimit
	}
	info, err := os.Stat(path)
	if err != nil {
		return "", err
	}
	if info.IsDir() {
		return "", fmt.Errorf("%s is a directory, not a file", path)
	}
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	buf := make([]byte, maxBytes)
	n, err := f.Read(buf)
	if n == 0 && err != nil && !errors.Is(err, io.EOF) {
		return "", err
	}
	result := map[string]any{
		"path":    path,
		"bytes":   info.Size(),
		"content": string(buf[:n]),
	}
	if info.Size() > int64(n) {
		result["truncated"] = true
		result["note"] = fmt.Sprintf("only the first %d bytes are shown", n)
	}
	return jsonResult(result)
}

func schema(kind, description string) map[string]any {
	return map[string]any{"type": kind, "description": description}
}

func jsonResult(v any) (string, error) {
	data, err := json.Marshal(v)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func argString(args map[string]any, key string) string {
	switch v := args[key].(type) {
	case string:
		return strings.TrimSpace(v)
	case nil:
		return ""
	default:
		return strings.TrimSpace(fmt.Sprintf("%v", v))
	}
}

func argInt(args map[string]any, key string, def int) int {
	switch v := args[key].(type) {
	case float64:
		return int(v)
	case int:
		return v
	case json.Number:
		if n, err := v.Int64(); err == nil {
			return int(n)
		}
	case string:
		if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil {
			return n
		}
	}
	return def
}

func argBool(args map[string]any, key string, def bool) bool {
	switch v := args[key].(type) {
	case bool:
		return v
	case string:
		if b, err := strconv.ParseBool(strings.TrimSpace(v)); err == nil {
			return b
		}
	}
	return def
}

func argStrings(args map[string]any, key string) []string {
	switch v := args[key].(type) {
	case []string:
		return v
	case []any:
		out := make([]string, 0, len(v))
		for _, item := range v {
			if s, ok := item.(string); ok {
				out = append(out, strings.TrimSpace(s))
			}
		}
		return out
	case string:
		var out []string
		for _, part := range strings.Split(v, ",") {
			if part = strings.TrimSpace(part); part != "" {
				out = append(out, part)
			}
		}
		return out
	}
	return nil
}
