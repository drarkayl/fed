package main

import (
	"fmt"
	"os"
	"os/signal"
	"strconv"
	"syscall"

	"github.com/sirupsen/logrus"
)

func main() {
	setupLogger()
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	port := p2pPort
	if len(os.Args) > 1 {
		logrus.Infof("Starting orchestrator with port: %v", os.Args[1])
		var err error
		port, err = strconv.Atoi(os.Args[1])
		if err != nil {
			logrus.Fatalf("Invalid port: %v", err)
		}
	} else {
		logrus.Infof("Starting orchestrator with default port %v", p2pPort)
	}
	logrus.AddHook(&customHook{nodeName: fmt.Sprint(port)}) // Add the hook

	logrus.Info("Go Orchestrator starting...")

	// Create a peer store
	peerStore := NewPeerStore()
	// Start the P2P server in a goroutine
	go StartP2PServer(peerStore, "go-orchestrator", port, quit)
	// Start discovering peers in a goroutine
	go DiscoverPeers(peerStore, "_p2p._tcp", quit)

	// // Create a new ML client
	// mlClient, err := NewMLClient()
	// if err != nil {
	// 	logrus.Fatalf("Failed to create ML client: %v", err)
	// }
	// defer mlClient.Close()

	// // Create a context with a timeout
	// ctx, cancel := context.WithTimeout(context.Background(), time.Second*10)
	// defer cancel()

	// // Perform ML operations
	// mlClient.TrainLocalModel(ctx)
	// mlClient.GetModelWeights(ctx)
	// mlClient.SetModelWeights(ctx)
	// mlClient.PerformInference(ctx)

	logrus.Info("Orchestrator running. Press Ctrl+C to exit.")

	// Wait for a signal to exit
	<-quit

	logrus.Info("Shutting down orchestrator...")
}
