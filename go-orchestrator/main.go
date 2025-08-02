package main

import (
	"context"
	"log"
	"time"

	pb "github.com/drarkayl/fed/proto/proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/types/known/emptypb"
)

const (
	mlServiceAddress = "localhost:50051"
)

func main() {
	// Set up a connection to the server.
	// Using WithTransportCredentials and insecure.NewCredentials() is the modern way to create an insecure connection.
	conn, err := grpc.NewClient(mlServiceAddress, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("did not connect: %v", err)
	}
	defer conn.Close()
	c := pb.NewMLServiceClient(conn)

	// Create a context with a timeout to prevent calls from hanging indefinitely.
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*10)
	defer cancel()

	// 1. Call TrainLocalModel
	log.Println("--- Calling TrainLocalModel ---")
	trainReq := &pb.TrainRequest{
		Epochs:    1,
		BatchSize: 32,
		RoundId:   1,
	}
	trainRes, err := c.TrainLocalModel(ctx, trainReq)
	if err != nil {
		log.Fatalf("could not train model: %v", err)
	}
	log.Printf("TrainLocalModel Response: Success=%v, Weights ClientID=%s", trainRes.GetConfirmation().GetSuccess(), trainRes.GetWeights().GetClientId())

	// 2. Call GetModelWeights
	log.Println("\n--- Calling GetModelWeights ---")
	weights, err := c.GetModelWeights(ctx, &emptypb.Empty{})
	if err != nil {
		log.Fatalf("could not get model weights: %v", err)
	}
	log.Printf("GetModelWeights Response: ClientID=%s, RoundID=%d", weights.GetClientId(), weights.GetRoundId())

	// 3. Call SetModelWeights
	log.Println("\n--- Calling SetModelWeights ---")
	setWeightsReq := &pb.SetGlobalWeightsRequest{
		GlobalWeights: &pb.ModelWeights{
			SerializedWeights: []byte("dummy-global-weights-from-go"),
			ClientId:          "go-orchestrator-client",
			RoundId:           1,
		},
	}
	setWeightsRes, err := c.SetModelWeights(ctx, setWeightsReq)
	if err != nil {
		log.Fatalf("could not set model weights: %v", err)
	}
	log.Printf("SetModelWeights Response: Success=%v, Message='%s'", setWeightsRes.GetSuccess(), setWeightsRes.GetMessage())

	// 4. Call PerformInference
	log.Println("\n--- Calling PerformInference ---")
	inferenceReq := &pb.ImageData{
		Data: []byte("dummy-image-data-from-go"),
	}
	inferenceRes, err := c.PerformInference(ctx, inferenceReq)
	if err != nil {
		log.Fatalf("could not perform inference: %v", err)
	}
	log.Printf("PerformInference Response: Label='%s', Confidence=%.2f", inferenceRes.GetLabel(), inferenceRes.GetConfidence())
}
