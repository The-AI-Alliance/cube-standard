// cube-sidecar — static exec server that runs inside a Toolkit container.
//
// Same HTTP protocol as _sidecar_server.py; drop-in replacement that needs
// no python3 in the container image.  Build with CGO_ENABLED=0 to produce a
// fully static binary that works in any Linux container (glibc, musl, scratch).
//
// Security posture matches the Python server — see _sidecar_server.py header.
package main

import (
	"bytes"
	"context"
	"crypto/hmac"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

const (
	maxBodyBytes = 1 * 1024 * 1024 // 1 MiB
	defaultPort  = "8787"
	bindAddr     = "127.0.0.1" // SECURITY: never 0.0.0.0
)

var token []byte

func loadToken() {
	path := os.Getenv("CUBE_SIDECAR_TOKEN_FILE")
	if path == "" {
		fmt.Fprintln(os.Stderr, "CUBE_SIDECAR_TOKEN_FILE not set")
		os.Exit(2)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to read token file: %v\n", err)
		os.Exit(2)
	}
	tok := strings.TrimSpace(string(data))
	if len(tok) < 32 {
		fmt.Fprintln(os.Stderr, "token too short")
		os.Exit(2)
	}
	token = []byte(tok)
}

func checkAuth(authHeader string) bool {
	if !strings.HasPrefix(authHeader, "Bearer ") {
		return false
	}
	presented := []byte(strings.TrimSpace(authHeader[len("Bearer "):]))
	return hmac.Equal(presented, token)
}

type execRequest struct {
	Command string            `json:"command"`
	Timeout float64           `json:"timeout"`
	Workdir string            `json:"workdir"`
	Env     map[string]string `json:"env"`
}

type execResponse struct {
	Stdout          string  `json:"stdout"`
	Stderr          string  `json:"stderr"`
	ExitCode        int     `json:"exit_code"`
	DurationSeconds float64 `json:"duration_seconds"`
}

func sendJSON(w http.ResponseWriter, status int, payload any) {
	body, _ := json.Marshal(payload)
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Length", strconv.Itoa(len(body)))
	w.WriteHeader(status)
	w.Write(body) //nolint:errcheck
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		sendJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method_not_allowed"})
		return
	}
	sendJSON(w, http.StatusOK, map[string]bool{"ok": true})
}

func handleExec(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		sendJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method_not_allowed"})
		return
	}
	if !checkAuth(r.Header.Get("Authorization")) {
		sendJSON(w, http.StatusUnauthorized, map[string]string{"error": "unauthorized"})
		return
	}

	raw, err := io.ReadAll(io.LimitReader(r.Body, maxBodyBytes+1))
	if err != nil || len(raw) > maxBodyBytes {
		sendJSON(w, http.StatusRequestEntityTooLarge, map[string]string{"error": "body_too_large"})
		return
	}

	var req execRequest
	req.Timeout = 120 // default
	if err := json.Unmarshal(raw, &req); err != nil {
		sendJSON(w, http.StatusBadRequest, map[string]string{"error": "bad_json"})
		return
	}
	if req.Command == "" {
		sendJSON(w, http.StatusBadRequest, map[string]string{"error": "bad_command"})
		return
	}
	if req.Timeout <= 0 || req.Timeout > 24*3600 {
		sendJSON(w, http.StatusBadRequest, map[string]string{"error": "bad_timeout"})
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(req.Timeout*float64(time.Second)))
	defer cancel()

	cmd := exec.CommandContext(ctx, "/bin/sh", "-c", req.Command)
	if req.Workdir != "" {
		cmd.Dir = req.Workdir
	}

	// Inherit env, strip sidecar token, apply overrides.
	env := os.Environ()
	filtered := make([]string, 0, len(env)+len(req.Env))
	for _, e := range env {
		if !strings.HasPrefix(e, "CUBE_SIDECAR_TOKEN_FILE=") {
			filtered = append(filtered, e)
		}
	}
	for k, v := range req.Env {
		filtered = append(filtered, k+"="+v)
	}
	cmd.Env = filtered

	var stdoutBuf, stderrBuf bytes.Buffer
	cmd.Stdout = &stdoutBuf
	cmd.Stderr = &stderrBuf

	start := time.Now()
	runErr := cmd.Run()
	duration := time.Since(start).Seconds()

	exitCode := 0
	if runErr != nil {
		if ctx.Err() != nil {
			exitCode = 124 // GNU timeout convention
			fmt.Fprintf(&stderrBuf, "\n[sidecar] timed out after %.0fs\n", req.Timeout)
		} else if exitErr, ok := runErr.(*exec.ExitError); ok {
			exitCode = exitErr.ExitCode()
		} else {
			exitCode = 1
		}
	}

	sendJSON(w, http.StatusOK, execResponse{
		Stdout:          stdoutBuf.String(),
		Stderr:          stderrBuf.String(),
		ExitCode:        exitCode,
		DurationSeconds: duration,
	})
}

func main() {
	loadToken()

	port := os.Getenv("CUBE_SIDECAR_PORT")
	if port == "" {
		port = defaultPort
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", handleHealth)
	mux.HandleFunc("/exec", handleExec)

	addr := net.JoinHostPort(bindAddr, port)
	fmt.Fprintf(os.Stderr, "cube-sidecar listening on %s\n", addr)

	srv := &http.Server{Addr: addr, Handler: mux}
	if err := srv.ListenAndServe(); err != nil {
		fmt.Fprintf(os.Stderr, "server error: %v\n", err)
		os.Exit(1)
	}
}
