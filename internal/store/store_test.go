package store

import (
	"os"
	"path/filepath"
	"testing"

	"micro-pa/internal/llm"
)

func newStore(t *testing.T) *Store {
	t.Helper()
	st, err := Open(t.TempDir())
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	return st
}

func TestAddAndSearchMemories(t *testing.T) {
	st := newStore(t)

	for _, text := range []string{
		"prefers tabs over spaces",
		"runs kubernetes in homelab",
		"drinks coffee in the morning",
	} {
		if _, err := st.AddMemory(text, nil); err != nil {
			t.Fatalf("AddMemory(%q): %v", text, err)
		}
	}

	hits, err := st.SearchMemories("coffee", 0)
	if err != nil {
		t.Fatalf("SearchMemories: %v", err)
	}
	if len(hits) != 1 || hits[0].Text != "drinks coffee in the morning" {
		t.Fatalf("search for coffee = %+v", hits)
	}

	// Multiple terms must all match.
	if hits, _ := st.SearchMemories("kubernetes homelab", 0); len(hits) != 1 {
		t.Errorf("expected one hit for both terms, got %d", len(hits))
	}
	if hits, _ := st.SearchMemories("kubernetes chicken", 0); len(hits) != 0 {
		t.Errorf("expected no hit when a term is missing, got %d", len(hits))
	}
	// An empty query returns the most recent notes first.
	all, _ := st.SearchMemories("", 0)
	if len(all) != 3 {
		t.Fatalf("empty query returned %d notes, want 3", len(all))
	}
	if all[0].Text != "drinks coffee in the morning" {
		t.Errorf("newest note first, got %q", all[0].Text)
	}
	// The limit is respected.
	if limited, _ := st.SearchMemories("", 1); len(limited) != 1 {
		t.Errorf("limit ignored: got %d", len(limited))
	}
}

func TestAddMemoryRejectsEmpty(t *testing.T) {
	st := newStore(t)
	if _, err := st.AddMemory("   ", nil); err == nil {
		t.Error("expected an error for an empty note")
	}
}

func TestForgetMemoryAndAmbiguousPrefix(t *testing.T) {
	st := newStore(t)
	seed := `[{"id":"m_aaa111","text":"first","created":"2026-01-01T00:00:00Z"},
              {"id":"m_aaa222","text":"second","created":"2026-01-01T00:00:00Z"}]`
	if err := os.WriteFile(filepath.Join(st.Dir(), "memory.json"), []byte(seed), 0o600); err != nil {
		t.Fatal(err)
	}

	// An ambiguous prefix must not delete anything.
	if _, ok, err := st.ForgetMemory("m_aaa"); err != nil || ok {
		t.Errorf("ambiguous prefix: ok=%v err=%v, want ok=false", ok, err)
	}
	if all, _ := st.Memories(); len(all) != 2 {
		t.Fatalf("ambiguous delete removed notes: %d left", len(all))
	}

	// An exact id deletes exactly one note.
	m, ok, err := st.ForgetMemory("m_aaa111")
	if err != nil || !ok || m.Text != "first" {
		t.Fatalf("ForgetMemory = %+v ok=%v err=%v", m, ok, err)
	}
	all, _ := st.Memories()
	if len(all) != 1 || all[0].ID != "m_aaa222" {
		t.Errorf("after delete: %+v", all)
	}

	// A missing id is reported, not an error.
	if _, ok, err := st.ForgetMemory("m_nope"); err != nil || ok {
		t.Errorf("missing id: ok=%v err=%v", ok, err)
	}
}

func TestTasks(t *testing.T) {
	st := newStore(t)
	first, err := st.AddTask("renew domain", "2026-10-04")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := st.AddTask("water plants", ""); err != nil {
		t.Fatal(err)
	}
	if _, err := st.AddTask("   ", ""); err == nil {
		t.Error("expected an error for an empty task")
	}

	tasks, err := st.Tasks()
	if err != nil {
		t.Fatal(err)
	}
	if len(tasks) != 2 {
		t.Fatalf("got %d tasks, want 2", len(tasks))
	}

	done, ok, err := st.SetTaskDone(first.ID, true)
	if err != nil || !ok || !done.Done || done.DoneAt == nil {
		t.Fatalf("SetTaskDone = %+v ok=%v err=%v", done, ok, err)
	}
	// Completed tasks sort after open ones.
	tasks, _ = st.Tasks()
	if tasks[len(tasks)-1].ID != first.ID {
		t.Errorf("completed task should sort last, got %+v", tasks)
	}
	// Reopening clears the completion time.
	reopened, _, _ := st.SetTaskDone(first.ID, false)
	if reopened.Done || reopened.DoneAt != nil {
		t.Errorf("reopened task still marked done: %+v", reopened)
	}
	if _, ok, err := st.SetTaskDone("t_missing", true); err != nil || ok {
		t.Errorf("missing task: ok=%v err=%v", ok, err)
	}
}

func TestSessions(t *testing.T) {
	st := newStore(t)
	msgs := []llm.Message{
		{Role: llm.RoleUser, Content: "hello"},
		{Role: llm.RoleAssistant, Content: "hi"},
	}
	if err := st.SaveSession("work", msgs); err != nil {
		t.Fatal(err)
	}
	if err := st.SaveSession("a/b weird name!", nil); err != nil {
		t.Fatal(err)
	}

	loaded, err := st.LoadSession("work")
	if err != nil {
		t.Fatal(err)
	}
	if len(loaded) != 2 || loaded[0].Content != "hello" {
		t.Errorf("LoadSession = %+v", loaded)
	}

	names, err := st.ListSessions()
	if err != nil {
		t.Fatal(err)
	}
	if len(names) != 2 {
		t.Fatalf("ListSessions = %v, want 2 entries", names)
	}
	for _, n := range names {
		if n == "a/b weird name!" {
			t.Errorf("session names must be sanitised, got %q", n)
		}
	}
	// A missing session loads as empty, not as an error.
	if msgs, err := st.LoadSession("never-existed"); err != nil || msgs != nil {
		t.Errorf("missing session: msgs=%v err=%v", msgs, err)
	}
}

func TestSaveLeavesNoTempFiles(t *testing.T) {
	st := newStore(t)
	if _, err := st.AddMemory("a note", []string{"x"}); err != nil {
		t.Fatal(err)
	}
	entries, err := os.ReadDir(st.Dir())
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		if len(e.Name()) > 0 && e.Name()[0] == '.' {
			t.Errorf("leftover temp file: %s", e.Name())
		}
	}
}

func TestLoadToleratesCorruptFile(t *testing.T) {
	st := newStore(t)
	if err := os.WriteFile(filepath.Join(st.Dir(), "memory.json"), []byte("{not json"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := st.Memories(); err == nil {
		t.Error("expected a parse error for a corrupt file")
	}
}
