# Go parameters
GOCMD=go
GODEP=dep
GOBUILD=$(GOCMD) build
GOCLEAN=$(GOCMD) clean
GOTEST=$(GOCMD) test
GOGET=$(GOCMD) get
CMDS=train predict
GOOS=linux
GOARCH=amd64

all: deps test $(CMDS)

%: cmd/%/main.go
	@echo "Input $<, Output $@"
	env GOOS=$(GOOS) GOARCH=$(GOARCH) $(GOBUILD) -v -o $@ $<

test: liblinear/*.go cmd/*/main.go
	$(GOTEST) -v ./...

clean: 
	$(GOCLEAN)
	rm -f $(CMDS)

deps:
	$(GODEP) ensure -v