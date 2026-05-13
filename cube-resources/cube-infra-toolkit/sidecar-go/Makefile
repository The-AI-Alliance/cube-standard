BINDIR := ../src/cube_infra_toolkit/_bin
LDFLAGS := -ldflags="-s -w"

.PHONY: all linux-amd64 linux-arm64 clean

all: linux-amd64 linux-arm64

linux-amd64:
	CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build $(LDFLAGS) -o $(BINDIR)/linux-amd64/cube-sidecar .

linux-arm64:
	CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build $(LDFLAGS) -o $(BINDIR)/linux-arm64/cube-sidecar .

clean:
	rm -f $(BINDIR)/linux-amd64/cube-sidecar $(BINDIR)/linux-arm64/cube-sidecar
