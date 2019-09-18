package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"flag"
	"fmt"
	"log"
	"math"
	"net/http"
	"strings"
	tf_core_framework "tensorflow/core/framework"
	pb "tensorflow_serving/apis"
	"time"

	"github.com/golang/protobuf/ptypes/empty"

	google_protobuf "github.com/golang/protobuf/ptypes/wrappers"
	"google.golang.org/grpc"

	videoService "raspividWrapper/videoService"
)

var (
	throttle                     = 1000 * time.Millisecond
	processMotionAddress         = "localhost:8500"
	modelName                    = "process_motion"
	signatureName                = "serving_default"
	modelVersion         int64   = 1
	inputName                    = "input"
	outputName                   = "output"
	totalAverageMotion   float32 = 5.
	motionGRPCAddress            = "mirror.local:5555"
	wakupAddress                 = "http://mirror.local:9080/"
	wakupThrottle                = 10 * time.Second
	mbx                  int64   = 104
	mby                  int64   = 78
)

func init() {
	flag.DurationVar(&throttle, "throttle", throttle, "throttle motion detection to save CPU")
	flag.DurationVar(&wakupThrottle, "wakeupthrottle", wakupThrottle, "minimum time before wakup signals are sent")
	flag.StringVar(&processMotionAddress, "servinggrpc", processMotionAddress, "GRPC address to TensorFlow Serving for Motion Processing")
	flag.StringVar(&modelName, "model", modelName, "TensorFlow Serving Model Name")
	flag.StringVar(&signatureName, "signature", signatureName, "TensorFlow Serving Model Signature Name")
	flag.Int64Var(&modelVersion, "modelversion", modelVersion, "TensorFlow Serving Model Version")
	flag.StringVar(&inputName, "modelinput", inputName, "TensorFlow Serving Model Input Name")
	flag.StringVar(&outputName, "modeloutput", outputName, "TensorFlow Serving Model Output Name")
	flag.StringVar(&motionGRPCAddress, "motiongrpc", motionGRPCAddress, "GRPC address of motion data")
	flag.StringVar(&modelName, "modelName", modelName, "name of motion processing model in tensorflow serving")
	flag.StringVar(&wakupAddress, "wakeupurl", wakupAddress, "http URL for posting wakup messages")
	flag.Int64Var(&mbx, "mbx", mbx, "macro blocks in X direction")
	flag.Int64Var(&mby, "mby", mby, "macro blocks in y direction")

}

type motionVector struct {
	X   int8
	Y   int8
	Sad int16
}

func serialCalculateMotionFroMVectors(body []byte, mbx, mby int) (float64, error) {
	log.Printf("len(body): %d", len(body))

	vect := make([]motionVector, mbx*mby)

	if err := binary.Read(bytes.NewReader(body), binary.LittleEndian, vect); err != nil {
		return 0., err
	}

	tot := 0.
	for i := 0; i < mbx*mby; i++ {
		x := float64(vect[i].X)
		y := float64(vect[i].Y)
		mag2 := x*x + y*y

		tot += math.Sqrt(mag2)
	}
	tot /= float64(mbx * mby)
	fmt.Printf("manually calculating total mag: %f\n", tot)

	return tot, nil
}

type MotionCalculator struct {
	ServingAddr   string
	ModelName     string
	SignatureName string
	Version       int64
	InputName     string
	OutputName    string
	MBX           int64
	MBY           int64

	conn   *grpc.ClientConn
	client pb.PredictionServiceClient
	req    *pb.PredictRequest
}

func (m *MotionCalculator) Open() error {
	var err error

	m.conn, err = grpc.Dial(processMotionAddress, grpc.WithInsecure())
	if err != nil {
		return err
	}

	m.client = pb.NewPredictionServiceClient(m.conn)

	m.req = &pb.PredictRequest{
		ModelSpec: &pb.ModelSpec{
			Name:          modelName,
			SignatureName: signatureName,
			Version: &google_protobuf.Int64Value{
				Value: modelVersion,
			},
		},
		Inputs: map[string]*tf_core_framework.TensorProto{
			inputName: &tf_core_framework.TensorProto{
				Dtype: tf_core_framework.DataType_DT_INT8,
				TensorShape: &tf_core_framework.TensorShapeProto{
					Dim: []*tf_core_framework.TensorShapeProto_Dim{
						&tf_core_framework.TensorShapeProto_Dim{Size: int64(1)},
						&tf_core_framework.TensorShapeProto_Dim{Size: int64(mbx)},
						&tf_core_framework.TensorShapeProto_Dim{Size: int64(mby)},
						&tf_core_framework.TensorShapeProto_Dim{Size: int64(4)},
					},
				},
			},
		},
	}
	return nil
}

func (m *MotionCalculator) Close() {
	m.conn.Close()
}

func (m *MotionCalculator) Calculate(body []byte) (float32, error) {
	m.req.Inputs[inputName].TensorContent = body

	res, err := m.client.Predict(context.Background(), m.req)
	if err != nil {
		return 0, err
	}

	o, ok := res.Outputs[outputName]
	if !ok {
		return 0, fmt.Errorf("output '%s' not found", outputName)
	}
	return o.GetFloatVal()[0], nil
}

func main() {
	flag.Parse()

	calc := &MotionCalculator{
		ServingAddr:   processMotionAddress,
		ModelName:     modelName,
		SignatureName: signatureName,
		Version:       modelVersion,
		InputName:     inputName,
		OutputName:    outputName,
		MBX:           mbx,
		MBY:           mby,
	}
	if err := calc.Open(); err != nil {
		log.Fatalf("error opening tf-serving connection: %v", err)
	}
	defer calc.Close()

	motConn, err := grpc.Dial(motionGRPCAddress, grpc.WithInsecure())
	if err != nil {
		log.Fatalf("error connecting to grpc motion address: %v", err)
	}

	motionClient := videoService.NewVideoClient(motConn)

	client, err := motionClient.MotionRaw(context.Background(), &empty.Empty{})
	if err != nil {
		log.Fatalf("error calling motionRaw: %v", err)
	}

	lastWakeup := time.Time{}

	for {
		t := time.Now()

		frame, err := client.Recv()
		if err != nil {
			log.Printf("error receiving frame: %v", err)
			break
		}

		body := frame.GetData()

		if motion, err := calc.Calculate(body); err != nil {
			log.Printf("error calculating motion: %v", err)
		} else if motion > totalAverageMotion {
			if time.Now().Sub(lastWakeup) > wakupThrottle {
				log.Printf("waking up")

				lastWakeup = time.Now()

				http.Post(wakupAddress, "application/json", strings.NewReader("\"on\""))
			}
		}

		if s := time.Now().Sub(t); s > throttle {
			time.Sleep(s)
		}
	}
}
