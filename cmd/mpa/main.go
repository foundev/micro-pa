// Command mpa is a small personal assistant that lives in your terminal.
//
// It talks to any OpenAI-compatible chat completions endpoint (OpenRouter by
// default), optionally calls a handful of local tools, and keeps its notes,
// tasks and conversations as plain JSON under ~/.micro-pa.
package main

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"micro-pa/internal/agent"
	"micro-pa/internal/config"
	"micro-pa/internal/llm"
	"micro-pa/internal/store"
	"micro-pa/internal/tools"
)

// version is the release string printed by -version.
const version = "0.1.0"

// maxHistory caps how many conversation messages are kept and replayed.
const maxHistory = 40

// app holds everything one run needs.
type app struct {
	cfg     *config.Config
	store   *store.Store
	tools   *tools.Registry
	agent   *agent.Agent
	client  llm.Client
	session string
	history []llm.Message
	// initialPrompt is a question given on the command line, asked before the
	// interactive prompt appears.
	initialPrompt string
	in            *bufio.Reader
	out           io.Writer
	trace         io.Writer
	confirm       func(string) bool
	verbose       bool
}

func main() { os.Exit(run(os.Args[1:], os.Stdin, os.Stdout, os.Stderr)) }

// run is main with its inputs passed in, which keeps it testable.
func run(args []string, stdin io.Reader, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("mpa", flag.ContinueOnError)
	fs.SetOutput(stderr)

	var (
		configPath   = fs.String("config", "", "path to config.json (default <home>/config.json)")
		provider     = fs.String("provider", "", "backend: openai (any OpenAI-compatible API) or mock (offline)")
		modelFlag    = fs.String("model", "", "model slug, for example deepseek/deepseek-v4-flash")
		baseURL      = fs.String("base-url", "", "API base URL, for example https://openrouter.ai/api/v1")
		apiKey       = fs.String("api-key", "", "API key (prefer MPA_API_KEY or OPENROUTER_API_KEY)")
		home         = fs.String("home", "", "state directory (default ~/.micro-pa)")
		persona      = fs.String("persona", "", "extra instructions about you and how micro-pa should behave")
		session      = fs.String("session", "", "conversation to continue (default \"default\")")
		once         = fs.String("once", "", "run one prompt and exit; use - to read the prompt from stdin")
		allowShell   = fs.Bool("allow-shell", false, "let the model run shell commands")
		initConfig   = fs.Bool("init", false, "write a starter config file and exit")
		listSessions = fs.Bool("sessions", false, "list saved conversations and exit")
		listTools    = fs.Bool("tools", false, "list available tools and exit")
		verbose      = fs.Bool("v", false, "print the system prompt at startup")
		showVersion  = fs.Bool("version", false, "print version and exit")
	)
	fs.Usage = func() { usage(fs, stderr) }

	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	fail := func(err error) int {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}

	onceSet := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "once" {
			onceSet = true
		}
	})

	if *showVersion {
		fmt.Fprintln(stdout, "micro-pa "+version)
		return 0
	}

	cfg, err := config.Load(*configPath)
	if err != nil {
		return fail(err)
	}
	if *provider != "" {
		cfg.Provider = *provider
	}
	if *modelFlag != "" {
		cfg.Model = *modelFlag
	}
	if *baseURL != "" {
		cfg.BaseURL = *baseURL
	}
	if *apiKey != "" {
		cfg.APIKey = *apiKey
	}
	if *home != "" {
		cfg.Home = *home
	}
	if *persona != "" {
		cfg.Persona = *persona
	}
	if *allowShell {
		cfg.AllowShell = true
	}
	cfg.Normalize()

	if *initConfig {
		if err := writeExampleConfig(cfg); err != nil {
			return fail(err)
		}
		fmt.Fprintf(stdout, "wrote %s\n", cfg.Path)
		return 0
	}

	st, err := store.Open(cfg.Home)
	if err != nil {
		return fail(err)
	}

	a := &app{
		cfg:     cfg,
		store:   st,
		in:      bufio.NewReader(stdin),
		out:     stdout,
		trace:   stderr,
		verbose: *verbose,
	}
	// In one-shot mode there is nobody to answer a confirmation prompt, so
	// shell commands run unattended (only when -allow-shell was passed).
	if !onceSet {
		a.confirm = a.confirmShell
	}
	if rest := fs.Args(); len(rest) > 0 && !onceSet && !*listTools && !*listSessions {
		a.initialPrompt = strings.Join(rest, " ")
	}
	if err := a.rebuildTools(); err != nil {
		return fail(err)
	}
	if err := a.setClient(); err != nil {
		return fail(err)
	}

	if *listTools {
		a.printTools()
		return 0
	}
	if *listSessions {
		a.printSessions()
		return 0
	}

	a.session = "default"
	if *session != "" {
		a.session = *session
	}
	history, err := st.LoadSession(a.session)
	if err != nil {
		fmt.Fprintln(stderr, "warning: could not load session:", err)
	}
	a.history = history

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if onceSet {
		prompt := *once
		if prompt == "-" {
			data, err := io.ReadAll(stdin)
			if err != nil {
				return fail(err)
			}
			prompt = string(data)
		}
		if strings.TrimSpace(prompt) == "" {
			return fail(errors.New("-once needs a prompt, or - to read one from stdin"))
		}
		if cfg.AllowShell {
			fmt.Fprintln(stderr, "note: shell tool is enabled and commands run without confirmation in -once mode")
		}
		return a.askOnce(ctx, prompt)
	}

	a.printBanner()
	if a.verbose {
		fmt.Fprintln(a.out, "\n--- system prompt ---")
		fmt.Fprintln(a.out, a.agent.System())
		fmt.Fprintln(a.out, "--- end system prompt ---")
	}
	return a.repl(ctx)
}

// clientBuilder, when set, replaces the default HTTP model client. Tests use it
// so they can exercise the CLI without opening a socket.
var clientBuilder func(*config.Config) llm.Client

// errNoKey is returned when the chosen backend needs a key that is missing.
var errNoKey = fmt.Errorf(`no API key configured for the openai backend.

Set one of these, then try again:

  export OPENROUTER_API_KEY=sk-or-...      # recommended for OpenRouter
  export MPA_API_KEY=sk-or-...             # same thing, micro-pa specific

Or start from a config file:

  mpa -init                                # writes a starter config.json

To try micro-pa offline, with no key and no network:

  mpa -provider mock`)

// setClient builds the model client and refreshes the agent.
func (a *app) setClient() error {
	if a.cfg.Provider == config.ProviderMock {
		a.client = llm.MockClient{}
	} else {
		if a.cfg.APIKey == "" {
			return errNoKey
		}
		if clientBuilder != nil {
			a.client = clientBuilder(a.cfg)
		} else {
			a.client = llm.NewOpenAIClient(
				a.cfg.BaseURL,
				a.cfg.APIKey,
				a.cfg.Model,
				a.cfg.Temperature,
				time.Duration(a.cfg.TimeoutSec)*time.Second,
			)
		}
	}
	a.agent = &agent.Agent{
		Client:   a.client,
		Tools:    a.tools,
		MaxTurns: a.cfg.MaxTurns,
		Persona:  a.cfg.Persona,
		Context:  a.contextForModel,
		Out:      a.trace,
	}
	return nil
}

// rebuildTools regenerates the tool set, for example after /shell toggles.
func (a *app) rebuildTools() error {
	reg, err := tools.Default(tools.Options{
		Store:      a.store,
		AllowShell: a.cfg.AllowShell,
		Confirm:    a.confirm,
	})
	if err != nil {
		return err
	}
	a.tools = reg
	if a.agent != nil {
		a.agent.Tools = reg
	}
	return nil
}

// contextForModel summarizes saved notes and open tasks for the system prompt,
// which is what makes micro-pa feel like it remembers across sessions.
func (a *app) contextForModel() string {
	var sb strings.Builder

	if mems, err := a.store.RecentMemories(8); err == nil && len(mems) > 0 {
		sb.WriteString("Recent notes (search the rest with recall):\n")
		for _, m := range mems {
			fmt.Fprintf(&sb, "- [%s] %s\n", m.ID, m.Text)
		}
	}

	if tasks, err := a.store.Tasks(); err == nil {
		var open []string
		for _, t := range tasks {
			if !t.Done {
				open = append(open, fmt.Sprintf("- [%s] %s%s", t.ID, t.Text, dueSuffix(t.Due)))
			}
		}
		if len(open) > 0 {
			fmt.Fprintf(&sb, "\nOpen tasks (%d):\n%s\n", len(open), strings.Join(open, "\n"))
		}
	}

	out := strings.TrimSpace(sb.String())
	if len(out) > 4000 {
		out = out[:4000] + "\n..."
	}
	return out
}

func dueSuffix(due string) string {
	if strings.TrimSpace(due) == "" {
		return ""
	}
	return " (due " + due + ")"
}

// askOnce answers a single prompt and saves the conversation.
func (a *app) askOnce(ctx context.Context, prompt string) int {
	history, err := a.agent.Ask(ctx, agent.Trim(a.history, maxHistory), prompt)
	if err != nil {
		return failTo(a.trace, err)
	}
	a.history = history
	if err := a.store.SaveSession(a.session, agent.Trim(a.history, maxHistory)); err != nil {
		fmt.Fprintln(a.trace, "warning: could not save session:", err)
	}
	if answer := agent.Answer(history); answer != "" {
		fmt.Fprintln(a.out, answer)
	}
	return 0
}

func writeExampleConfig(cfg *config.Config) error {
	if _, err := os.Stat(cfg.Path); err == nil {
		return fmt.Errorf("%s already exists, not overwriting", cfg.Path)
	}
	example := *cfg
	example.APIKey = ""
	if example.Persona == "" {
		example.Persona = "Keep answers short. I mostly work in Go, on Linux."
	}
	return example.Save()
}

func failTo(w io.Writer, err error) int {
	fmt.Fprintln(w, "error:", err)
	return 1
}

func usage(fs *flag.FlagSet, w io.Writer) {
	fmt.Fprintf(w, `micro-pa %s - a small personal assistant in your terminal

Usage:
  mpa [flags] [question]    start an interactive session (optionally asking
                            the question first)
  mpa -once "question"      answer one question and exit
  echo "question" | mpa -once -

Backend:
  model       -model      default %s
  base URL    -base-url   default %s
  API key     MPA_API_KEY or OPENROUTER_API_KEY in the environment
  offline     -provider mock    no key, no network, canned replies

Examples:
  export OPENROUTER_API_KEY=sk-or-...
  mpa "what is on my list?"                 # asks once, then stays interactive
  mpa -once "summarize the notes I saved about kubernetes"
  mpa -session work -allow-shell            # shell tool on, confirms each command
  mpa -persona "I prefer metric units"      # extra system instructions

Flags:
`, version, config.DefaultModel, config.DefaultBaseURL)
	fs.PrintDefaults()
}
