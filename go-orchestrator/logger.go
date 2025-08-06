package main

import (
	"os"

	"github.com/sirupsen/logrus"
)

type customHook struct {
	nodeName string
}

func (h *customHook) Levels() []logrus.Level {
	return logrus.AllLevels
}

func (h *customHook) Fire(entry *logrus.Entry) error {
	entry.Data["node_port"] = h.nodeName // your default field and value
	return nil
}

func setupLogger() {
	// Configure logrus to output in JSON format.
	logrus.SetFormatter(&logrus.JSONFormatter{})

	// Configure logrus to write to a file.
	file, err := os.OpenFile("go-orchestrator.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
	if err == nil {
		logrus.SetOutput(file)
	} else {
		logrus.Info("Failed to log to file, using default stderr")
	}
}
