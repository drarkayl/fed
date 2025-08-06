package main

import (
	"context"

	pb "github.com/drarkayl/fed/proto/proto"
	"github.com/sirupsen/logrus"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/types/known/emptypb"
)

// MLClient is a wrapper around the MLServiceClient
type MLClient struct {
	conn *grpc.ClientConn
	c    pb.MLServiceClient
}

// NewMLClient creates a new MLClient
func NewMLClient() (*MLClient, error) {
	conn, err := grpc.NewClient(mlServiceAddress, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, err
	}
	c := pb.NewMLServiceClient(conn)
	return &MLClient{conn: conn, c: c}, nil
}

// Close closes the connection to the ML service
func (c *MLClient) Close() {
	c.conn.Close()
}

// TrainLocalModel calls the TrainLocalModel RPC
func (c *MLClient) TrainLocalModel(ctx context.Context) {
	logrus.Info("--- Calling TrainLocalModel ---")
	trainReq := &pb.TrainRequest{
		Epochs:    1,
		BatchSize: 32,
		RoundId:   1,
	}
	trainRes, err := c.c.TrainLocalModel(ctx, trainReq)
	if err != nil {
		logrus.WithFields(logrus.Fields{"error": err}).Fatal("Could not train model")
	}
	logrus.WithFields(logrus.Fields{
		"success":  trainRes.GetConfirmation().GetSuccess(),
		"clientID": trainRes.GetWeights().GetClientId(),
	}).Info("TrainLocalModel Response")
}

// GetModelWeights calls the GetModelWeights RPC
func (c *MLClient) GetModelWeights(ctx context.Context) {
	logrus.Info("--- Calling GetModelWeights ---")
	weights, err := c.c.GetModelWeights(ctx, &emptypb.Empty{})
	if err != nil {
		logrus.WithFields(logrus.Fields{"error": err}).Fatal("Could not get model weights")
	}
	logrus.WithFields(logrus.Fields{
		"clientID": weights.GetClientId(),
		"roundID":  weights.GetRoundId(),
	}).Info("GetModelWeights Response")
}

// SetModelWeights calls the SetModelWeights RPC
func (c *MLClient) SetModelWeights(ctx context.Context) {
	logrus.Info("--- Calling SetModelWeights ---")
	setWeightsReq := &pb.SetGlobalWeightsRequest{
		GlobalWeights: &pb.ModelWeights{
			SerializedWeights: []byte("dummy-global-weights-from-go"),
			ClientId:          "go-orchestrator-client",
			RoundId:           1,
		},
	}
	setWeightsRes, err := c.c.SetModelWeights(ctx, setWeightsReq)
	if err != nil {
		logrus.WithFields(logrus.Fields{"error": err}).Fatal("Could not set model weights")
	}
	logrus.WithFields(logrus.Fields{
		"success": setWeightsRes.GetSuccess(),
		"message": setWeightsRes.GetMessage(),
	}).Info("SetModelWeights Response")
}

// PerformInference calls the PerformInference RPC
func (c *MLClient) PerformInference(ctx context.Context) {
	logrus.Info("--- Calling PerformInference ---")
	inferenceReq := &pb.ImageData{
		Data: []byte("dummy-image-data-from-go"),
	}
	inferenceRes, err := c.c.PerformInference(ctx, inferenceReq)
	if err != nil {
		logrus.WithFields(logrus.Fields{"error": err}).Fatal("Could not perform inference")
	}
	logrus.WithFields(logrus.Fields{
		"label":      inferenceRes.GetLabel(),
		"confidence": inferenceRes.GetConfidence(),
	}).Info("PerformInference Response")
}
