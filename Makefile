BIN := mpa
PKG := ./cmd/mpa

.PHONY: build test race vet fmt check demo clean

build:
	go build -o $(BIN) $(PKG)

test:
	go test ./...

race:
	go test -race ./...

vet:
	go vet ./...

fmt:
	gofmt -w .

# What CI should run.
check: fmt vet test

# A self-contained demo: no key, no network.
demo: build
	MPA_HOME=$(CURDIR)/.demo ./$(BIN) -provider mock -init
	MPA_HOME=$(CURDIR)/.demo ./$(BIN) -provider mock -once "remember that I prefer tabs"
	MPA_HOME=$(CURDIR)/.demo ./$(BIN) -provider mock -once "what time is it"

clean:
	rm -f $(BIN)
	rm -rf .demo
