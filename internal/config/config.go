// Package config holds the settings for a micro-pa run.
//
// Precedence, lowest to highest:
//
//	defaults < $MPA_HOME/config.json < environment variables < command line flags
package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

const (
	// ProviderOpenAI talks to any OpenAI-compatible /chat/completions endpoint.
	ProviderOpenAI = "openai"
	// ProviderMock is fully offline and never touches the network.
	ProviderMock = "mock"

	// DefaultBaseURL is OpenRouter, the default backend for micro-pa.
	DefaultBaseURL = "https://openrouter.ai/api/v1"
	// LocalBaseURL is a typical local server (vLLM, llama.cpp, LM Studio).
	LocalBaseURL = "http://localhost:8000/v1"
	// DefaultModel is an OpenRouter alias that tracks the current DeepSeek
	// Flash release. Pin an exact slug (e.g. deepseek/deepseek-v4-flash-0731)
	// in config.json when you want a frozen version.
	DefaultModel = "~deepseek/deepseek-flash-latest"

	// DefaultMaxTurns caps how many tool-calling rounds one question may use.
	DefaultMaxTurns = 8
	// DefaultTimeoutSec is the per-request HTTP timeout.
	DefaultTimeoutSec = 120
	// DefaultTemperature is used when the config does not set one.
	DefaultTemperature = 0.7
)

// Config is the full configuration for one micro-pa process.
type Config struct {
	Provider    string  `json:"provider"`
	BaseURL     string  `json:"base_url"`
	APIKey      string  `json:"api_key"`
	Model       string  `json:"model"`
	Home        string  `json:"home"`
	Persona     string  `json:"persona"`
	AllowShell  bool    `json:"allow_shell"`
	MaxTurns    int     `json:"max_turns"`
	TimeoutSec  int     `json:"timeout_seconds"`
	Temperature float64 `json:"temperature"`

	// Path records where the config file was read from, if any. It is never
	// written back to disk.
	Path string `json:"-"`
}

// Default returns the configuration used when nothing else is configured.
func Default() *Config {
	return &Config{
		Provider:    ProviderOpenAI,
		BaseURL:     DefaultBaseURL,
		Model:       DefaultModel,
		Home:        DefaultHome(),
		MaxTurns:    DefaultMaxTurns,
		TimeoutSec:  DefaultTimeoutSec,
		Temperature: DefaultTemperature,
	}
}

// DefaultHome returns ~/.micro-pa, or $MPA_HOME when set.
func DefaultHome() string {
	if v := strings.TrimSpace(os.Getenv("MPA_HOME")); v != "" {
		return expandHome(v)
	}
	dir, err := os.UserHomeDir()
	if err != nil {
		return ".micro-pa"
	}
	return filepath.Join(dir, ".micro-pa")
}

// Load reads the config file and then applies environment overrides. A missing
// explicitPath is an error; a missing default path simply means "use defaults".
func Load(explicitPath string) (*Config, error) {
	cfg := Default()

	path := explicitPath
	if path == "" {
		path = filepath.Join(cfg.Home, "config.json")
	}
	path = expandHome(path)
	cfg.Path = path

	data, err := os.ReadFile(path)
	switch {
	case err == nil:
		if err := json.Unmarshal(data, cfg); err != nil {
			return nil, fmt.Errorf("parse %s: %w", path, err)
		}
	case os.IsNotExist(err) && explicitPath != "":
		return nil, fmt.Errorf("config file not found: %s", path)
	case os.IsNotExist(err):
		// No config file yet: defaults plus environment are enough.
	default:
		return nil, fmt.Errorf("read %s: %w", path, err)
	}

	cfg.applyEnv()
	cfg.Normalize()
	return cfg, nil
}

// Save writes the config to disk with owner-only permissions.
func (c *Config) Save() error {
	if err := os.MkdirAll(filepath.Dir(c.Path), 0o700); err != nil {
		return err
	}
	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(c.Path, append(data, '\n'), 0o600)
}

func (c *Config) applyEnv() {
	setString(&c.Provider, "MPA_PROVIDER")
	setString(&c.BaseURL, "MPA_BASE_URL")
	setString(&c.Model, "MPA_MODEL")
	setString(&c.Home, "MPA_HOME")
	setString(&c.Persona, "MPA_PERSONA")

	if v := firstSet("MPA_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"); v != "" {
		c.APIKey = v
	}
	if v := os.Getenv("OPENAI_BASE_URL"); v != "" && os.Getenv("MPA_BASE_URL") == "" {
		c.BaseURL = v
	}
	if v := os.Getenv("MPA_ALLOW_SHELL"); v != "" {
		c.AllowShell = parseBool(v)
	}
	if v := os.Getenv("MPA_MAX_TURNS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			c.MaxTurns = n
		}
	}
	if v := os.Getenv("MPA_TIMEOUT_SECONDS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			c.TimeoutSec = n
		}
	}
	if v := os.Getenv("MPA_TEMPERATURE"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			c.Temperature = f
		}
	}
}

// Normalize fills in defaults for anything left empty after a file, the
// environment, or command line flags have been applied.
func (c *Config) Normalize() {
	c.Provider = strings.ToLower(strings.TrimSpace(c.Provider))
	if c.Provider == "" {
		c.Provider = ProviderOpenAI
	}
	c.Home = expandHome(strings.TrimSpace(c.Home))
	if c.Home == "" {
		c.Home = DefaultHome()
	}
	c.BaseURL = strings.TrimRight(strings.TrimSpace(c.BaseURL), "/")
	if c.BaseURL == "" && c.Provider == ProviderOpenAI {
		c.BaseURL = DefaultBaseURL
	}
	if strings.TrimSpace(c.Model) == "" && c.Provider == ProviderOpenAI {
		c.Model = DefaultModel
	}
	switch {
	case c.MaxTurns <= 0:
		c.MaxTurns = DefaultMaxTurns
	case c.MaxTurns > 32:
		c.MaxTurns = 32
	}
	if c.TimeoutSec <= 0 {
		c.TimeoutSec = DefaultTimeoutSec
	}
	if c.Temperature <= 0 {
		c.Temperature = DefaultTemperature
	}
}

// NeedsKey reports whether the active provider needs an API key to work.
func (c *Config) NeedsKey() bool {
	return c.Provider == ProviderOpenAI && c.APIKey == ""
}

// Summary is a one-line, secret-free description of the active setup.
func (c *Config) Summary() string {
	if c.Provider == ProviderMock {
		return "mock provider (offline, no network)"
	}
	return fmt.Sprintf("%s via %s (temp %.1f)", c.Model, c.BaseURL, c.Temperature)
}

// Redacted renders the config as JSON with the API key masked, for /config.
func (c *Config) Redacted() string {
	clone := *c
	if clone.APIKey != "" {
		clone.APIKey = "***" + lastN(clone.APIKey, 4)
	}
	data, err := json.MarshalIndent(&clone, "", "  ")
	if err != nil {
		return err.Error()
	}
	return string(data)
}

func expandHome(p string) string {
	if p == "~" || strings.HasPrefix(p, "~/") {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, strings.TrimPrefix(strings.TrimPrefix(p, "~"), "/"))
		}
	}
	return p
}

func setString(dst *string, key string) {
	if v, ok := os.LookupEnv(key); ok && strings.TrimSpace(v) != "" {
		*dst = v
	}
}

func firstSet(keys ...string) string {
	for _, k := range keys {
		if v := os.Getenv(k); strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}

func parseBool(v string) bool {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "1", "true", "yes", "y", "on":
		return true
	default:
		return false
	}
}

func lastN(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[len(s)-n:]
}
