// Package store keeps micro-pa state on disk: notes, tasks and saved sessions.
//
// Everything is a plain JSON file under one directory (~/.micro-pa by default) so
// that the whole state is inspectable, greppable and easy to back up.
package store

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"micro-pa/internal/llm"
)

// Memory is one saved note.
type Memory struct {
	ID      string    `json:"id"`
	Text    string    `json:"text"`
	Tags    []string  `json:"tags,omitempty"`
	Created time.Time `json:"created"`
}

// Task is one todo item.
type Task struct {
	ID      string     `json:"id"`
	Text    string     `json:"text"`
	Done    bool       `json:"done"`
	Due     string     `json:"due,omitempty"`
	Created time.Time  `json:"created"`
	DoneAt  *time.Time `json:"done_at,omitempty"`
}

// Session is a saved conversation.
type Session struct {
	Name     string        `json:"name"`
	Updated  time.Time     `json:"updated"`
	Messages []llm.Message `json:"messages"`
}

// Store is a JSON-file-backed state directory.
type Store struct {
	dir string
	mu  sync.Mutex
}

// Open prepares the state directory, creating it if needed.
func Open(dir string) (*Store, error) {
	if err := os.MkdirAll(filepath.Join(dir, "sessions"), 0o700); err != nil {
		return nil, fmt.Errorf("create %s: %w", dir, err)
	}
	return &Store{dir: dir}, nil
}

// Dir returns the directory holding all state.
func (s *Store) Dir() string { return s.dir }

func (s *Store) path(name string) string { return filepath.Join(s.dir, name) }

// Load decodes a JSON file into v. A missing or empty file is not an error.
func (s *Store) Load(name string, v any) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.loadLocked(name, v)
}

func (s *Store) loadLocked(name string, v any) error {
	data, err := os.ReadFile(s.path(name))
	switch {
	case os.IsNotExist(err):
		return nil
	case err != nil:
		return err
	}
	if len(strings.TrimSpace(string(data))) == 0 {
		return nil
	}
	if err := json.Unmarshal(data, v); err != nil {
		return fmt.Errorf("parse %s: %w", name, err)
	}
	return nil
}

// Save writes v as pretty JSON. The write is atomic: a temp file is renamed
// into place, so a crash can never leave a half-written state file.
func (s *Store) Save(name string, v any) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.saveLocked(name, v)
}

func (s *Store) saveLocked(name string, v any) error {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	full := s.path(name)
	dir := filepath.Dir(full)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".tmp-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err := tmp.Write(append(data, '\n')); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, full)
}

// Memories returns every stored note, oldest first.
func (s *Store) Memories() ([]Memory, error) {
	var out []Memory
	if err := s.Load("memory.json", &out); err != nil {
		return nil, err
	}
	return out, nil
}

// AddMemory appends a note and returns it.
func (s *Store) AddMemory(text string, tags []string) (Memory, error) {
	text = strings.TrimSpace(text)
	if text == "" {
		return Memory{}, fmt.Errorf("cannot remember an empty note")
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	var all []Memory
	if err := s.loadLocked("memory.json", &all); err != nil {
		return Memory{}, err
	}
	m := Memory{ID: NewID("m"), Text: text, Tags: cleanTags(tags), Created: time.Now()}
	all = append(all, m)
	return m, s.saveLocked("memory.json", all)
}

// ForgetMemory deletes a note by id (exact, or an unambiguous prefix).
func (s *Store) ForgetMemory(id string) (Memory, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var all []Memory
	if err := s.loadLocked("memory.json", &all); err != nil {
		return Memory{}, false, err
	}
	idx := matchIndex(len(all), func(i int) string { return all[i].ID }, id)
	if idx < 0 {
		return Memory{}, false, nil
	}
	removed := all[idx]
	all = append(all[:idx], all[idx+1:]...)
	return removed, true, s.saveLocked("memory.json", all)
}

// SearchMemories finds notes containing every term in the query, newest first.
// An empty query returns the most recent notes.
func (s *Store) SearchMemories(query string, limit int) ([]Memory, error) {
	all, err := s.Memories()
	if err != nil {
		return nil, err
	}
	if limit <= 0 {
		limit = 8
	}
	terms := strings.Fields(strings.ToLower(query))

	type scored struct {
		m     Memory
		score int
	}
	var hits []scored
	for _, m := range all {
		haystack := strings.ToLower(m.Text + " " + strings.Join(m.Tags, " "))
		if len(terms) == 0 {
			hits = append(hits, scored{m: m, score: 1})
			continue
		}
		score, matchedAll := 0, true
		for _, term := range terms {
			n := strings.Count(haystack, term)
			if n == 0 {
				matchedAll = false
				break
			}
			score += n
		}
		if matchedAll {
			hits = append(hits, scored{m: m, score: score})
		}
	}
	sort.SliceStable(hits, func(i, j int) bool {
		if hits[i].score != hits[j].score {
			return hits[i].score > hits[j].score
		}
		return hits[i].m.Created.After(hits[j].m.Created)
	})
	if len(hits) > limit {
		hits = hits[:limit]
	}
	out := make([]Memory, 0, len(hits))
	for _, h := range hits {
		out = append(out, h.m)
	}
	return out, nil
}

// RecentMemories returns up to n notes, newest first.
func (s *Store) RecentMemories(n int) ([]Memory, error) {
	all, err := s.Memories()
	if err != nil {
		return nil, err
	}
	if len(all) > n {
		all = all[len(all)-n:]
	}
	for i, j := 0, len(all)-1; i < j; i, j = i+1, j-1 {
		all[i], all[j] = all[j], all[i]
	}
	return all, nil
}

// Tasks returns every task, open ones first.
func (s *Store) Tasks() ([]Task, error) {
	var out []Task
	if err := s.Load("tasks.json", &out); err != nil {
		return nil, err
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Done != out[j].Done {
			return !out[i].Done
		}
		return out[i].Created.Before(out[j].Created)
	})
	return out, nil
}

// AddTask appends a task and returns it.
func (s *Store) AddTask(text, due string) (Task, error) {
	text = strings.TrimSpace(text)
	if text == "" {
		return Task{}, fmt.Errorf("cannot add an empty task")
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	var all []Task
	if err := s.loadLocked("tasks.json", &all); err != nil {
		return Task{}, err
	}
	t := Task{ID: NewID("t"), Text: text, Due: strings.TrimSpace(due), Created: time.Now()}
	all = append(all, t)
	return t, s.saveLocked("tasks.json", all)
}

// SetTaskDone marks a task done (or open again) and returns the updated task.
func (s *Store) SetTaskDone(id string, done bool) (Task, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var all []Task
	if err := s.loadLocked("tasks.json", &all); err != nil {
		return Task{}, false, err
	}
	idx := matchIndex(len(all), func(i int) string { return all[i].ID }, id)
	if idx < 0 {
		return Task{}, false, nil
	}
	all[idx].Done = done
	if done {
		now := time.Now()
		all[idx].DoneAt = &now
	} else {
		all[idx].DoneAt = nil
	}
	return all[idx], true, s.saveLocked("tasks.json", all)
}

// SaveSession writes a conversation to sessions/<name>.json.
func (s *Store) SaveSession(name string, messages []llm.Message) error {
	return s.Save(filepath.Join("sessions", safeName(name)+".json"), Session{
		Name:     name,
		Updated:  time.Now(),
		Messages: messages,
	})
}

// LoadSession reads a conversation. A missing session is an empty one.
func (s *Store) LoadSession(name string) ([]llm.Message, error) {
	var sess Session
	if err := s.Load(filepath.Join("sessions", safeName(name)+".json"), &sess); err != nil {
		return nil, err
	}
	return sess.Messages, nil
}

// ListSessions returns saved session names, most recently updated first.
func (s *Store) ListSessions() ([]string, error) {
	entries, err := os.ReadDir(filepath.Join(s.dir, "sessions"))
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	type named struct {
		name    string
		updated time.Time
	}
	var found []named
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		var sess Session
		full := filepath.Join(s.dir, "sessions", e.Name())
		if data, err := os.ReadFile(full); err == nil {
			_ = json.Unmarshal(data, &sess) // best effort: a corrupt file just sorts last
		}
		found = append(found, named{
			name:    strings.TrimSuffix(e.Name(), ".json"),
			updated: sess.Updated,
		})
	}
	sort.SliceStable(found, func(i, j int) bool { return found[i].updated.After(found[j].updated) })
	out := make([]string, 0, len(found))
	for _, f := range found {
		out = append(out, f.name)
	}
	return out, nil
}

// NewID returns a short, readable, collision-resistant id like m_9f2a41.
func NewID(prefix string) string {
	buf := make([]byte, 3)
	if _, err := rand.Read(buf); err != nil {
		return fmt.Sprintf("%s_%d", prefix, time.Now().UnixNano())
	}
	return prefix + "_" + hex.EncodeToString(buf)
}

// matchIndex finds the index of an id, accepting an unambiguous prefix.
func matchIndex(n int, idAt func(int) string, want string) int {
	want = strings.TrimSpace(want)
	if want == "" {
		return -1
	}
	exact, prefix, count := -1, -1, 0
	for i := 0; i < n; i++ {
		id := idAt(i)
		if id == want {
			exact = i
		}
		if strings.HasPrefix(id, want) {
			prefix = i
			count++
		}
	}
	if exact >= 0 {
		return exact
	}
	if count == 1 {
		return prefix
	}
	return -1
}

func cleanTags(tags []string) []string {
	out := make([]string, 0, len(tags))
	for _, t := range tags {
		if t = strings.TrimSpace(t); t != "" {
			out = append(out, t)
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func safeName(name string) string {
	name = strings.TrimSpace(name)
	if name == "" {
		return "default"
	}
	return strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
			return r
		case r == '-', r == '_', r == '.':
			return r
		default:
			return '-'
		}
	}, name)
}
