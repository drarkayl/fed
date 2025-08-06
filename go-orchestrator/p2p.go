package main

import (
	"context"
	"fmt"
	"net"
	"os"
	"sync"
	"time"

	pb "github.com/drarkayl/fed/proto/proto"
	"github.com/sirupsen/logrus"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type Peer struct {
	ID   string
	Port int
	Conn *grpc.ClientConn
}

type PeerStore struct {
	mu    sync.RWMutex
	peers map[string]*Peer
}

func NewPeerStore() *PeerStore {
	return &PeerStore{
		peers: make(map[string]*Peer),
	}
}

// Add adds a new peer to the store.
func (ps *PeerStore) Add(id string, port int) {
	ps.mu.Lock()
	defer ps.mu.Unlock()

	if _, ok := ps.peers[id]; ok {
		return // Peer already exists
	}

	conn, err := grpc.NewClient(id, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		logrus.WithFields(logrus.Fields{"error": err, "peerAddr": id}).Error("Failed to connect to peer")
		return
	}

	ps.peers[id] = &Peer{ID: id, Port: port, Conn: conn}
	logrus.Infof("Added peer %s at %s", id, id)
}

// Get returns a peer from the store.
func (ps *PeerStore) Get(id string) (*Peer, bool) {
	ps.mu.RLock()
	defer ps.mu.RUnlock()
	peer, ok := ps.peers[id]
	return peer, ok
}

// List returns all peers in the store.
func (ps *PeerStore) List() []*Peer {
	ps.mu.RLock()
	defer ps.mu.RUnlock()
	peers := make([]*Peer, 0, len(ps.peers))
	for _, peer := range ps.peers {
		peers = append(peers, peer)
	}
	return peers
}

// P2PService is our implementation of the P2PServiceServer interface
type P2PService struct {
	pb.UnimplementedP2PServiceServer
}

// SubmitLocalWeights is a stub implementation. temporarily used for testing purposes.
func (s *P2PService) SubmitLocalWeights(ctx context.Context, in *pb.ModelWeights) (*pb.AckResponse, error) {
	logrus.WithFields(logrus.Fields{
		"clientID": in.GetClientId(),
		"roundID":  in.GetRoundId(),
	}).Info("P2P: Received local weights")
	return &pb.AckResponse{Success: true, Message: "Weights received"}, nil
}

// BroadcastGlobalModel is a stub implementation. temporarily used for testing purposes.
func (s *P2PService) BroadcastGlobalModel(ctx context.Context, in *pb.ModelWeights) (*pb.AckResponse, error) {
	logrus.WithFields(logrus.Fields{
		"clientID": in.GetClientId(),
		"roundID":  in.GetRoundId(),
	}).Info("P2P: Received global model")
	return &pb.AckResponse{Success: true, Message: "Global model received"}, nil
}

// StartP2PServer starts the gRPC server for P2P communication and registers it with mDNS.
func StartP2PServer(peerStore *PeerStore, host string, port int, stopChan chan os.Signal) {
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		logrus.Fatalf("failed to listen: %v", err)
	}

	// Create a new gRPC server
	s := grpc.NewServer()
	pb.RegisterP2PServiceServer(s, &P2PService{})

	go func() {
		// Wait for the stop signal
		<-stopChan
		logrus.Info("gRPC server received stop signal. Performing graceful shutdown...")
		s.GracefulStop()
	}()

	// // Start serving
	if err := s.Serve(lis); err != nil {
		logrus.Fatalf("failed to serve: %v", err)
	}

}

// to be added StartTraining function

// DiscoverPeers uses mDNS to continuously discover other orchestrator nodes.
func DiscoverPeers(peerStore *PeerStore, serviceTag string, stopChan chan os.Signal) {
	// A channel to signal that the probe has finished
	probeDone := make(chan struct{})

	// Start the first discovery probe immediately in a goroutine
	logrus.Info("Starting new peer discovery...")
	go DiscoveryProbe(peerStore, probeDone)

	for {
		select {
		case <-probeDone:
			// The previous probe is finished. Wait for 30 seconds, then start a new one.
			logrus.Info("Peer discovery round complete. Waiting 30 seconds for next round...")
			time.Sleep(30 * time.Second)
			logrus.Info("Starting new peer discovery...")
			go DiscoveryProbe(peerStore, probeDone)

		case <-stopChan:
			logrus.Info("Stopping discovery...")
			return
		}
	}
}

// DiscoveryProbe sends discovery probes to a range of ports.
// It uses a WaitGroup to ensure all probes are sent before signaling completion.
func DiscoveryProbe(peerStore *PeerStore, doneChan chan struct{}) {
	var wg sync.WaitGroup
	logrus.Info("Browse for P2P peers...")
	for i := 50000; i <= 50100; i++ {
		wg.Add(1)
		go func(port int) {
			defer wg.Done()
			address := fmt.Sprintf("localhost:%d", port)
			conn, err := net.DialTimeout("tcp", address, 1*time.Second)
			if err != nil {
				return
			}
			defer conn.Close()
			logrus.Info("Found peer at ", address)
			peerStore.Add(address, port)
		}(i)
	}

	wg.Wait()              // Wait for all probe goroutines to finish
	doneChan <- struct{}{} // Signal that the entire probe is complete
}
