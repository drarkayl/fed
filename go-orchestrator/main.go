package main

import (
	"context"
	"os"
	"time"

	pb "github.com/drarkayl/fed/proto/proto"
	"github.com/sirupsen/logrus"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/types/known/emptypb"
)

const (
	mlServiceAddress = "localhost:50051"
)

func main() {
	// --- Structured Logging Setup ---
	// Configure logrus to output in JSON format.
	logrus.SetFormatter(&logrus.JSONFormatter{})

	// Configure logrus to write to a file.
	file, err := os.OpenFile("go-orchestrator.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
	if err == nil {
		logrus.SetOutput(file)
	} else {
		logrus.Info("Failed to log to file, using default stderr")
	}
	defer file.Close()

	logrus.Info("Go Orchestrator client starting...")

	// Set up a connection to the server.
	// Using WithTransportCredentials and insecure.NewCredentials() is the modern way to create an insecure connection.
	conn, err := grpc.NewClient(mlServiceAddress, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		logrus.WithFields(logrus.Fields{"error": err}).Fatal("Failed to connect to gRPC server")
	}
	defer conn.Close()
	c := pb.NewMLServiceClient(conn)

	// Create a context with a timeout to prevent calls from hanging indefinitely.
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*10)
	defer cancel()

	// 1. Call TrainLocalModel
	logrus.Info("--- Calling TrainLocalModel ---")
	trainReq := &pb.TrainRequest{
		Epochs:    1,
		BatchSize: 32,
		RoundId:   1,
	}
	trainRes, err := c.TrainLocalModel(ctx, trainReq)
	if err != nil {
		logrus.WithFields(logrus.Fields{"error": err}).Fatal("Could not train model")
	}
	logrus.WithFields(logrus.Fields{
		"success":  trainRes.GetConfirmation().GetSuccess(),
		"clientID": trainRes.GetWeights().GetClientId(),
	}).Info("TrainLocalModel Response")

	// 2. Call GetModelWeights
	logrus.Info("--- Calling GetModelWeights ---")
	weights, err := c.GetModelWeights(ctx, &emptypb.Empty{})
	if err != nil {
		logrus.WithFields(logrus.Fields{"error": err}).Fatal("Could not get model weights")
	}
	logrus.WithFields(logrus.Fields{
		"clientID": weights.GetClientId(),
		"roundID":  weights.GetRoundId(),
	}).Info("GetModelWeights Response")

	// 3. Call SetModelWeights
	logrus.Info("--- Calling SetModelWeights ---")
	setWeightsReq := &pb.SetGlobalWeightsRequest{
		GlobalWeights: &pb.ModelWeights{
			SerializedWeights: []byte("dummy-global-weights-from-go"),
			ClientId:          "go-orchestrator-client",
			RoundId:           1,
		},
	}
	setWeightsRes, err := c.SetModelWeights(ctx, setWeightsReq)
	if err != nil {
		logrus.WithFields(logrus.Fields{"error": err}).Fatal("Could not set model weights")
	}
	logrus.WithFields(logrus.Fields{
		"success": setWeightsRes.GetSuccess(),
		"message": setWeightsRes.GetMessage(),
	}).Info("SetModelWeights Response")

	// 4. Call PerformInference
	logrus.Info("--- Calling PerformInference ---")
	inferenceReq := &pb.ImageData{
		Data: []byte("dummy-image-data-from-go"),
	}
	inferenceRes, err := c.PerformInference(ctx, inferenceReq)
	if err != nil {
		logrus.WithFields(logrus.Fields{"error": err}).Fatal("Could not perform inference")
	}
	logrus.WithFields(logrus.Fields{
		"label":      inferenceRes.GetLabel(),
		"confidence": inferenceRes.GetConfidence(),
	}).Info("PerformInference Response")
}
