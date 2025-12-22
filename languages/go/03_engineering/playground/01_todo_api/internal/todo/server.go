package todo

import (
	"context"
	"errors"
	"log"
	"net/http"
	"sync"
)

// Server represents the Todo API skeleton with dependency hooks.
type Server struct {
	once   sync.Once
	server *http.Server
}

// NewServer wires default HTTP server without handlers yet.
func NewServer() *Server {
	return &Server{}
}

// Start boots the HTTP server and returns immediately for now.
func (s *Server) Start() error {
	s.once.Do(func() {
		s.server = &http.Server{Addr: ":0"}
	})

	log.Println("todo api placeholder ready; add handlers before serving traffic")
	return nil
}

// Shutdown demonstrates how we will stop the server once Start blocks.
func (s *Server) Shutdown(ctx context.Context) error {
	if s.server == nil {
		return errors.New("server not started")
	}

	return s.server.Shutdown(ctx)
}
