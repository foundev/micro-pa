package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"

	"micro-pa/internal/agent"
)

// repl runs the interactive loop until the user quits or stdin closes.
func (a *app) repl(ctx context.Context) int {
	if a.initialPrompt != "" {
		a.turn(ctx, a.initialPrompt)
	}
	for {
		if err := ctx.Err(); err != nil {
			fmt.Fprintln(a.out, "\ninterrupted")
			return 130
		}
		line, err := a.prompt("you > ")
		if err != nil {
			fmt.Fprintln(a.out)
			return 0
		}
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "/") {
			if quit := a.command(ctx, line); quit {
				return 0
			}
			continue
		}
		a.turn(ctx, line)
	}
}

// prompt prints a prompt and reads one line. It reports io.EOF when stdin ends.
func (a *app) prompt(label string) (string, error) {
	fmt.Fprint(a.out, label)
	line, err := a.in.ReadString('\n')
	if err != nil {
		if errors.Is(err, io.EOF) && strings.TrimSpace(line) != "" {
			return line, nil
		}
		return "", err
	}
	return line, nil
}

// turn sends one user message, prints the answer and saves the session.
func (a *app) turn(ctx context.Context, input string) {
	history, err := a.agent.Ask(ctx, agent.Trim(a.history, maxHistory), input)
	if err != nil {
		if errors.Is(err, context.Canceled) {
			fmt.Fprintln(a.out, "cancelled")
			return
		}
		fmt.Fprintln(a.out, "error:", err)
		return
	}
	a.history = history
	if err := a.store.SaveSession(a.session, agent.Trim(a.history, maxHistory)); err != nil {
		fmt.Fprintln(a.trace, "warning: could not save session:", err)
	}
	if answer := agent.Answer(history); answer != "" {
		fmt.Fprintln(a.out, answer)
	}
	fmt.Fprintln(a.out)
}

// command handles a slash command. It returns true when the user wants to quit.
func (a *app) command(ctx context.Context, line string) bool {
	fields := strings.Fields(line)
	name := strings.ToLower(strings.TrimPrefix(fields[0], "/"))
	rest := strings.TrimSpace(strings.TrimPrefix(line, fields[0]))
	if name == "" {
		return false
	}

	switch name {
	case "quit", "exit", "q":
		return true

	case "help", "h", "?":
		a.printHelp()

	case "tools":
		a.printTools()

	case "memory", "memories", "notes":
		a.printMemories(rest)

	case "remember":
		if rest == "" {
			fmt.Fprintln(a.out, "usage: /remember <text>")
			return false
		}
		m, err := a.store.AddMemory(rest, []string{"manual"})
		if err != nil {
			fmt.Fprintln(a.out, "error:", err)
			return false
		}
		fmt.Fprintf(a.out, "saved %s\n", m.ID)

	case "forget", "forget-memory":
		if rest == "" {
			fmt.Fprintln(a.out, "usage: /forget <id>")
			return false
		}
		m, ok, err := a.store.ForgetMemory(rest)
		if err != nil {
			fmt.Fprintln(a.out, "error:", err)
			return false
		}
		if !ok {
			fmt.Fprintf(a.out, "no note with id %q\n", rest)
			return false
		}
		fmt.Fprintf(a.out, "forgot %s: %s\n", m.ID, m.Text)

	case "tasks":
		a.printTasks(strings.EqualFold(rest, "all"))

	case "done":
		if rest == "" {
			fmt.Fprintln(a.out, "usage: /done <id>")
			return false
		}
		t, ok, err := a.store.SetTaskDone(rest, true)
		if err != nil {
			fmt.Fprintln(a.out, "error:", err)
			return false
		}
		if !ok {
			fmt.Fprintf(a.out, "no task with id %q\n", rest)
			return false
		}
		fmt.Fprintf(a.out, "done: %s\n", t.Text)

	case "new":
		a.history = nil
		if err := a.store.SaveSession(a.session, nil); err != nil {
			fmt.Fprintln(a.out, "error:", err)
		}
		fmt.Fprintf(a.out, "started a fresh conversation in session %q\n", a.session)

	case "sessions":
		a.printSessions()

	case "session":
		if rest == "" {
			fmt.Fprintf(a.out, "current session: %s\n", a.session)
			return false
		}
		if err := a.switchSession(rest); err != nil {
			fmt.Fprintln(a.out, "error:", err)
		}

	case "model":
		if rest == "" {
			fmt.Fprintf(a.out, "model: %s\n", a.cfg.Model)
			return false
		}
		a.cfg.Model = rest
		if err := a.setClient(); err != nil {
			fmt.Fprintln(a.out, "error:", err)
			return false
		}
		fmt.Fprintf(a.out, "model: %s\n", a.cfg.Model)

	case "shell":
		a.toggleShell(rest)

	case "config":
		fmt.Fprintln(a.out, a.cfg.Redacted())

	default:
		fmt.Fprintf(a.out, "unknown command /%s. Try /help.\n", name)
	}
	return false
}

// toggleShell turns the shell tool on or off. Turning it on is a deliberate,
// visible step because it lets the model touch the machine.
func (a *app) toggleShell(arg string) {
	switch strings.ToLower(arg) {
	case "on", "true", "yes":
		a.cfg.AllowShell = true
	case "off", "false", "no":
		a.cfg.AllowShell = false
	case "":
		a.cfg.AllowShell = !a.cfg.AllowShell
	default:
		fmt.Fprintln(a.out, "usage: /shell on|off")
		return
	}
	if err := a.rebuildTools(); err != nil {
		fmt.Fprintln(a.out, "error:", err)
		return
	}
	if a.cfg.AllowShell {
		fmt.Fprintln(a.out, "shell tool enabled; you will be asked to approve each command")
	} else {
		fmt.Fprintln(a.out, "shell tool disabled")
	}
}

// switchSession saves the current conversation and opens another.
func (a *app) switchSession(name string) error {
	if err := a.store.SaveSession(a.session, agent.Trim(a.history, maxHistory)); err != nil {
		return err
	}
	history, err := a.store.LoadSession(name)
	if err != nil {
		return err
	}
	a.session = name
	a.history = history
	count := 0
	for _, m := range history {
		if m.Role == "user" {
			count++
		}
	}
	fmt.Fprintf(a.out, "session %q (%d previous questions)\n", name, count)
	return nil
}

// confirmShell asks the user before a command runs.
func (a *app) confirmShell(command string) bool {
	fmt.Fprintf(a.out, "\n  run this command? %s\n  [y/N] ", command)
	line, err := a.in.ReadString('\n')
	if err != nil && strings.TrimSpace(line) == "" {
		fmt.Fprintln(a.out)
		return false
	}
	switch strings.ToLower(strings.TrimSpace(line)) {
	case "y", "yes":
		return true
	default:
		fmt.Fprintln(a.out, "  declined")
		return false
	}
}

func (a *app) printBanner() {
	fmt.Fprintf(a.out, "micro-pa %s - %s\n", version, a.cfg.Summary())
	fmt.Fprintf(a.out, "session %q, state in %s\n", a.session, a.store.Dir())

	var notes []string
	if mems, err := a.store.RecentMemories(1); err == nil && len(mems) > 0 {
		if all, err := a.store.Memories(); err == nil {
			notes = append(notes, fmt.Sprintf("%d notes", len(all)))
		}
	}
	if tasks, err := a.store.Tasks(); err == nil {
		open := 0
		for _, t := range tasks {
			if !t.Done {
				open++
			}
		}
		if open > 0 {
			notes = append(notes, fmt.Sprintf("%d open tasks", open))
		}
	}
	if len(notes) > 0 {
		fmt.Fprintf(a.out, "remembering %s\n", strings.Join(notes, ", "))
	}
	if a.cfg.AllowShell {
		fmt.Fprintln(a.out, "shell tool is on")
	}
	fmt.Fprintln(a.out, "type /help for commands, /quit to leave")
	fmt.Fprintln(a.out)
}

func (a *app) printHelp() {
	fmt.Fprint(a.out, `Commands
  /help                 this list
  /tools                tools the model can call
  /memory [query]       saved notes, optionally filtered
  /remember <text>      save a note yourself
  /forget <id>          delete a note
  /tasks [all]          open tasks, or all of them
  /done <id>            complete a task
  /new                  start a fresh conversation
  /sessions             list saved conversations
  /session <name>       switch to another conversation
  /model [slug]         show or change the model
  /shell on|off         enable the shell tool (asks before each command)
  /config               show the active settings
  /quit                 leave

Anything else is sent to the model.`)
	fmt.Fprintln(a.out)
}

func (a *app) printTools() {
	names := a.tools.Names()
	if len(names) == 0 {
		fmt.Fprintln(a.out, "no tools available")
		return
	}
	fmt.Fprintf(a.out, "%d tools:\n", len(names))
	for _, n := range names {
		fmt.Fprintf(a.out, "  %s\n", n)
	}
	if !a.cfg.AllowShell {
		fmt.Fprintln(a.out, "\nshell is off; enable it with /shell on")
	}
}

func (a *app) printMemories(query string) {
	found, err := a.store.SearchMemories(query, 20)
	if err != nil {
		fmt.Fprintln(a.out, "error:", err)
		return
	}
	if len(found) == 0 {
		fmt.Fprintln(a.out, "nothing saved yet")
		return
	}
	for _, m := range found {
		tags := ""
		if len(m.Tags) > 0 {
			tags = " (" + strings.Join(m.Tags, ", ") + ")"
		}
		fmt.Fprintf(a.out, "  %s  %s%s\n", m.ID, m.Text, tags)
	}
}

func (a *app) printTasks(all bool) {
	tasks, err := a.store.Tasks()
	if err != nil {
		fmt.Fprintln(a.out, "error:", err)
		return
	}
	shown := 0
	for _, t := range tasks {
		if t.Done && !all {
			continue
		}
		box := "[ ]"
		if t.Done {
			box = "[x]"
		}
		due := ""
		if t.Due != "" {
			due = "  due " + t.Due
		}
		fmt.Fprintf(a.out, "  %s %s  %s%s\n", box, t.ID, t.Text, due)
		shown++
	}
	if shown == 0 {
		fmt.Fprintln(a.out, "no open tasks")
	}
}

func (a *app) printSessions() {
	names, err := a.store.ListSessions()
	if err != nil {
		fmt.Fprintln(a.out, "error:", err)
		return
	}
	if len(names) == 0 {
		fmt.Fprintln(a.out, "no saved conversations yet")
		return
	}
	for _, n := range names {
		marker := "  "
		if n == a.session {
			marker = "* "
		}
		fmt.Fprintf(a.out, "%s%s\n", marker, n)
	}
}
