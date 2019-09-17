package main

import (
	"context"
	"log"
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
	motionURL                    = "http://mirror.local:8888/motion.bin"
	processMotionAddress         = "localhost:8500"
	totalAverageMotion   float32 = 5.
	motionGRPCAddress            = "mirror.local:5555"
	wakupAddress                 = "http://mirror.local:9080/"
	wakupThrottle                = 10 * time.Second
)

type motionVector struct {
	X   int8
	Y   int8
	Sad int16
}

func calculateMotionFromVectors(predictClient pb.PredictionServiceClient, body []byte) (float32, error) {
	// log.Printf("len(body): %d", len(body))

	// vect := make([]motionVector, 104*78)

	// err := binary.Read(bytes.NewReader(body), binary.LittleEndian, vect)

	// tot := 0.
	// for i := 0; i < 104*78; i++ {
	// 	x := float64(vect[i].X)
	// 	y := float64(vect[i].Y)
	// 	mag2 := x*x + y*y

	// 	tot += math.Sqrt(mag2)
	// }
	// tot /= (104 * 78)
	// fmt.Printf("manually calculating total mag: %f\n", tot)

	request := &pb.PredictRequest{
		ModelSpec: &pb.ModelSpec{
			Name:          "process_motion",
			SignatureName: "serving_default",
			Version: &google_protobuf.Int64Value{
				Value: int64(1),
			},
		},
		Inputs: map[string]*tf_core_framework.TensorProto{
			"input": &tf_core_framework.TensorProto{
				Dtype: tf_core_framework.DataType_DT_INT8,
				TensorShape: &tf_core_framework.TensorShapeProto{
					Dim: []*tf_core_framework.TensorShapeProto_Dim{
						&tf_core_framework.TensorShapeProto_Dim{Size: int64(1)},
						&tf_core_framework.TensorShapeProto_Dim{Size: int64(104)},
						&tf_core_framework.TensorShapeProto_Dim{Size: int64(78)},
						&tf_core_framework.TensorShapeProto_Dim{Size: int64(4)},
					},
				},
				TensorContent: body,
			},
		},
	}

	res, err := predictClient.Predict(context.Background(), request)
	if err != nil {
		return 0, err
	}

	return res.Outputs["output"].GetFloatVal()[0], nil
}

func main() {
	conn, err := grpc.Dial(processMotionAddress, grpc.WithInsecure())
	if err != nil {
		log.Fatalf("error connecting to serving: %v", err)
	}
	defer conn.Close()

	predictClient := pb.NewPredictionServiceClient(conn)

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

		if motion, err := calculateMotionFromVectors(predictClient, body); err != nil {
			log.Printf("error calculating motion: %v", err)
		} else if motion > totalAverageMotion {
			if time.Now().Sub(lastWakeup) > wakupThrottle {
				log.Printf("waking up")

				lastWakeup = time.Now()

				http.Post(wakupAddress, "application/json", strings.NewReader("\"on\""))
			}
		}

		// res, err := http.Get(motionURL)
		// if err != nil {
		// 	log.Printf("error retrieving motion data from '%s': %v", motionURL, err)
		// } else if res.StatusCode != http.StatusOK {
		// 	body, _ := ioutil.ReadAll(res.Body)

		// 	log.Printf("non-ok status (%s) code from url: '%s': %s", res.Status, motionURL, string(body))
		// } else if body, err := ioutil.ReadAll(res.Body); err != nil {
		// 	log.Printf("error reading body: %v", err)
		// } else if motion, err := calculateMotionFromVectors(predictClient, body); err != nil {
		// 	log.Printf("error calculating motion: %v", err)
		// } else {
		// 	fmt.Printf("*** MOTION: %f\n", motion)
		// }

		if s := time.Now().Sub(t); s > 0 {
			time.Sleep(s)
		}
	}
}
