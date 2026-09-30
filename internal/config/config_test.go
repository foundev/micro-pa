package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// isolate points MPA_HOME at a temp dir and clears every variable Load reads.
func isolate(t *testing.T, extra map[string]string) string {
	t.Helper()
	home := t.TempDir()
	for _, key := range []string{
		"MPA_PROVIDER", "MPA_BASE_URL", "MPA_MODEL", "MPA_HOME", "MPA_PERSONA",
		"MPA_API_KEY", "MPA_ALLOW_SHELL", "MPA_MAX_TURNS", "MPA_TIMEOUT_SECONDS",
		"MPA_TEMPERATURE", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL",
	} {
		t.Setenv(key, "")
	}
	t.Setenv("MPA_HOME", home)
	for k, v := range extra {
		t.Setenv(k, v)
	}
	return home
}

func TestDefaults(t *testing.T) {
	isolate(t, nil)
	cfg, err := Load("")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Provider != ProviderOpenAI {
		t.Errorf("provider = %q", cfg.Provider)
	}
	if cfg.Model != DefaultModel {
		t.Errorf("model = %q, want %q", cfg.Model, DefaultModel)
	}
	if cfg.BaseURL != DefaultBaseURL {
		t.Errorf("base url = %q", cfg.BaseURL)
	}
	if cfg.MaxTurns != DefaultMaxTurns || cfg.TimeoutSec != DefaultTimeoutSec {
		t.Errorf("limits = %d/%d", cfg.MaxTurns, cfg.TimeoutSec)
	}
	if !cfg.NeedsKey() {
		t.Error("the openai provider should report that it needs a key")
	}
}

func TestFileThenEnvironmentThenFlags(t *testing.T) {
	home := isolate(t, map[string]string{"MPA_MODEL": "from-env"})

	// A config file is written first.
	body := map[string]any{"model": "from-file", "max_turns": 3, "temperature": 0.2}
	data, _ := json.Marshal(body)
	if err := os.WriteFile(filepath.Join(home, "config.json"), data, 0o600); err != nil {
		t.Fatal(err)
	}

	cfg, err := Load("")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Model != "from-env" {
		t.Errorf("environment should beat the file, got model %q", cfg.Model)
	}
	if cfg.MaxTurns != 3 {
		t.Errorf("file value should survive, got max_turns %d", cfg.MaxTurns)
	}

	// The caller then applies command line flags, which win over everything.
	cfg.Model = "from-flag"
	cfg.Normalize()
	if cfg.Model != "from-flag" {
		t.Errorf("flag should win, got %q", cfg.Model)
	}
	if cfg.Temperature != 0.2 {
		t.Errorf("temperature = %v, want 0.2", cfg.Temperature)
	}
}

func TestMissingExplicitConfigIsAnError(t *testing.T) {
	isolate(t, nil)
	if _, err := Load(filepath.Join(t.TempDir(), "nope.json")); err == nil {
		t.Error("expected an error for a missing explicit config path")
	}
}

func TestBadConfigIsReported(t *testing.T) {
	home := isolate(t, nil)
	if err := os.WriteFile(filepath.Join(home, "config.json"), []byte("{oops"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(""); err == nil {
		t.Error("expected a parse error")
	}
}

func TestNormalizeClampsAndFills(t *testing.T) {
	cfg := &Config{Provider: "  MOCK ", MaxTurns: 999, TimeoutSec: -5, Temperature: 0}
	cfg.Normalize()
	if cfg.Provider != ProviderMock {
		t.Errorf("provider = %q, want mock", cfg.Provider)
	}
	if cfg.MaxTurns != 32 {
		t.Errorf("max turns should be clamped to 32, got %d", cfg.MaxTurns)
	}
	if cfg.TimeoutSec != DefaultTimeoutSec {
		t.Errorf("timeout = %d", cfg.TimeoutSec)
	}
	if cfg.Temperature != DefaultTemperature {
		t.Errorf("temperature = %v", cfg.Temperature)
	}
	if cfg.Home == "" {
		t.Error("home should be filled in")
	}
	// The mock provider does not need a key or a model.
	if cfg.NeedsKey() {
		t.Error("the mock provider should not need a key")
	}
}

func TestKeyFallbacksAndRedaction(t *testing.T) {
	isolate(t, map[string]string{"OPENROUTER_API_KEY": "sk-or-secret-abcd"})
	cfg, err := Load("")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.APIKey != "sk-or-secret-abcd" {
		t.Errorf("OPENROUTER_API_KEY should be picked up, got %q", cfg.APIKey)
	}
	if cfg.NeedsKey() {
		t.Error("a key is present, NeedsKey should be false")
	}
	redacted := cfg.Redacted()
	if strings.Contains(redacted, "secret") {
		t.Errorf("the key leaked into Redacted output:\n%s", redacted)
	}
	if !strings.Contains(redacted, "***abcd") {
		t.Errorf("Redacted should keep a short suffix for identification:\n%s", redacted)
	}
}

func TestSaveAndReload(t *testing.T) {
	home := isolate(t, nil)
	cfg, err := Load("")
	if err != nil {
		t.Fatal(err)
	}
	cfg.Persona = "I prefer metric units"
	cfg.AllowShell = true
	if err := cfg.Save(); err != nil {
		t.Fatal(err)
	}

	info, err := os.Stat(filepath.Join(home, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0o600 {
		t.Errorf("config permissions = %v, want 0600", info.Mode().Perm())
	}

	reloaded, err := Load("")
	if err != nil {
		t.Fatal(err)
	}
	if reloaded.Persona != "I prefer metric units" || !reloaded.AllowShell {
		t.Errorf("reloaded config lost values: %+v", reloaded)
	}
}

func TestExpandHome(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skip("no home directory in this environment")
	}
	if got := expandHome("~/notes"); got != filepath.Join(home, "notes") {
		t.Errorf("expandHome = %q", got)
	}
	if got := expandHome("/absolute"); got != "/absolute" {
		t.Errorf("expandHome left an absolute path alone: %q", got)
	}
}
