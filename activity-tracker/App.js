import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, TouchableOpacity, View, Button, Alert } from 'react-native';
import { Accelerometer, Gyroscope } from 'expo-sensors';
import {registerCalibration, registerMeasurement, resetCalibration, resetModel, startCalibration} from "./networking";

const BUFFER_LEN = 128
const LABELS = [
  "WALKING",
  "WALKING_UPSTAIRS",
  "WALKING_DOWNSTAIRS",
  "SITTING",
  "STANDING",
  "LAYING"
]


export default function App() {
  const [data, setData] = useState({
    acc_x: Array(BUFFER_LEN).fill(0.0),
    acc_y: Array(BUFFER_LEN).fill(0.0),
    acc_z: Array(BUFFER_LEN).fill(0.0),
    gyro_x: Array(BUFFER_LEN).fill(0.0),
    gyro_y: Array(BUFFER_LEN).fill(0.0),
    gyro_z: Array(BUFFER_LEN).fill(0.0),
  });

  const [predResponse, setPredResponse] = useState({
    pred: null,
    prob: null,
    pred_time: null
  })
  const [accSubscription, setAccSubscription] = useState(null);
  const [gyroSubscription, setGyroSubscription] = useState(null);
  const [buttonState, setButtonState] = useState(null);

  const _subscribe = () => {
    Accelerometer.setUpdateInterval(20);
    Gyroscope.setUpdateInterval(20);

    setAccSubscription(
        Accelerometer.addListener(accelerometerData => {
          const { x, y, z } = accelerometerData;
          setData(prev => {
            const acc_x = [...prev.acc_x]
            acc_x.pop()
            acc_x.unshift(x)

            const acc_y = [...prev.acc_y]
            acc_y.pop()
            acc_y.unshift(y)

            const acc_z = [...prev.acc_z]
            acc_z.pop()
            acc_z.unshift(z)

            return {...prev, acc_x, acc_y, acc_z}
          })
        })
    );
    setGyroSubscription(
        Gyroscope.addListener(gyroscopeData => {
          const { x, y, z } = gyroscopeData;
          setData(prev => {
            const newData = {...prev}
            const gyro_x = [...prev.gyro_x]
            gyro_x.pop()
            gyro_x.unshift(x)

            const gyro_y = [...prev.gyro_y]
            gyro_y.pop()
            gyro_y.unshift(y)

            const gyro_z = [...prev.gyro_z]
            gyro_z.pop()
            gyro_z.unshift(z)

            return {...prev, gyro_x, gyro_y, gyro_z}
          })
        })
    );
  };

  useEffect(() => {
    _subscribe();
  }, []);


  let registerFn = () => {}
  switch (buttonState) {
    case null:
      registerFn = registerMeasurement;
      break;
    case 'WALKING':
      registerFn = (data) => registerCalibration(data, 'WALKING')
      break;
    case 'WALKING_UPSTAIRS':
      registerFn = (data) => registerCalibration(data, 'WALKING_UPSTAIRS')
      break;
    case 'WALKING_DOWNSTAIRS':
      registerFn = (data) => registerCalibration(data, 'WALKING_DOWNSTAIRS')
      break;
    case 'SITTING':
      registerFn = (data) => registerCalibration(data, 'SITTING')
      break;
    case 'STANDING':
      registerFn = (data) => registerCalibration(data, 'STANDING')
      break;
    case 'LAYING':
      registerFn = (data) => registerCalibration(data, 'LAYING')
      break;
    default:
      registerFn = registerMeasurement;
      break;
  }

  const [count, setCount] = useState(0);
  useEffect(() => {
    const interval = setInterval(() => {
      registerFn(data).then(res => setPredResponse(res))
      setCount(count + 1);
    }, 64*20);
    return () => clearInterval(interval)
  }, [count]);

  const { acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z } = data;
  const { pred, prob, pred_time } = predResponse;

  return (
      <View style={styles.container}>
        <Text style={{...styles.text, fontSize: 16, fontWeight: "500"}}>Current activity</Text>
        <Text style={styles.text}>
          acc_x: {round(acc_x[BUFFER_LEN-1])} acc_y: {round(acc_y[BUFFER_LEN-1])} acc_z: {round(acc_z[BUFFER_LEN-1])}
        </Text>
        <Text style={styles.text}>
          gyro_x: {round(gyro_x[BUFFER_LEN-1])} gyro_y: {round(gyro_y[BUFFER_LEN-1])} gyro_z: {round(gyro_z[BUFFER_LEN-1])}
        </Text>
        <Text style={styles.text}>
          activity: {pred} confidence: {round(prob)} time: {round(pred_time)}
        </Text>
        <View style={{marginTop:80, marginLeft: 10, marginRight: 10}}>
          <Text style={{...styles.text, fontSize: 16, fontWeight: "500"}}>Track activity to fine tune the AI</Text>
          <View style={{flexDirection: 'row', marginTop: 10, justifyContent: 'center', flexWrap: 'wrap',  marginLeft: 30, marginRight: 30, marginBottom: 20}}>
            <TouchableOpacity
                onPress={() => setButtonState('WALKING')}
                style={ buttonState === 'WALKING' ? {
                  borderWidth:1,
                  borderColor:'#007AFF',
                  alignItems:'center',
                  justifyContent:'center',
                  width:80,
                  height:80,
                  backgroundColor:'#007AFF10',
                  borderRadius:50,
                  margin: 10
                } : {
                  borderWidth:1,
                  borderColor:'lightgrey',
                  alignItems:'center',
                  justifyContent:'center',
                  width:80,
                  height:80,
                  borderRadius:50,
                  margin: 10
                }}
            >
              <Text color={buttonState === 'WALKING' ? '#007AFF' : "grey"}>Walking</Text>
            </TouchableOpacity>
            <TouchableOpacity
                onPress={() => setButtonState('STANDING')}
                style={ buttonState === 'STANDING' ? {
                  borderWidth:1,
                  borderColor:'#007AFF',
                  alignItems:'center',
                  justifyContent:'center',
                  width:80,
                  height:80,
                  backgroundColor:'#007AFF10',
                  borderRadius:50,
                  margin: 10
                } : {
                  borderWidth:1,
                  borderColor:'lightgrey',
                  alignItems:'center',
                  justifyContent:'center',
                  width:80,
                  height:80,
                  borderRadius:50,
                  margin: 10
                }}
            >
              <Text color={buttonState === 'STANDING' ? '#007AFF' : "grey"}>Standing</Text>
            </TouchableOpacity>
            <TouchableOpacity
                onPress={() => setButtonState('SITTING')}
                style={ buttonState === 'SITTING' ? {
                  borderWidth:1,
                  borderColor:'#007AFF',
                  alignItems:'center',
                  justifyContent:'center',
                  width:80,
                  height:80,
                  backgroundColor:'#007AFF10',
                  borderRadius:50,
                  margin: 10
                } : {
                  borderWidth:1,
                  borderColor:'lightgrey',
                  alignItems:'center',
                  justifyContent:'center',
                  width:80,
                  height:80,
                  borderRadius:50,
                  margin: 10
                }}
            >
              <Text color={buttonState === 'SITTING' ? '#007AFF' : "grey"}>Sitting</Text>
            </TouchableOpacity>
            <TouchableOpacity
                onPress={() => setButtonState('LAYING')}
                style={ buttonState === 'LAYING' ? {
                  borderWidth:1,
                  borderColor:'#007AFF',
                  alignItems:'center',
                  justifyContent:'center',
                  width:80,
                  height:80,
                  backgroundColor:'#007AFF10',
                  borderRadius:50,
                  margin: 10
                } : {
                  borderWidth:1,
                  borderColor:'lightgrey',
                  alignItems:'center',
                  justifyContent:'center',
                  width:80,
                  height:80,
                  borderRadius:50,
                  margin: 10
                }}
            >
              <Text color={buttonState === 'LAYING' ? '#007AFF' : "grey"}>Laying</Text>
            </TouchableOpacity>
          </View>
          <Button title='Stop tracking' onPress={() => setButtonState(null)}/>
          <View style={{marginTop:10}}>
            <Button title='Reset measurements' color='red' onPress={() => resetCalibration().then(res => Alert.alert("Cleared measurements"))}/>
          </View>
        </View>
        <View style={{marginTop: 50, justifyContent: 'center', flexDirection: 'row'}}>
          <TouchableOpacity
              onPress={() => startCalibration()}
              style={{
                alignItems:'center',
                justifyContent:'center',
                width:180,
                height:50,
                backgroundColor:'#007AFFAF',
                borderRadius:11,
                margin: 10,
              }}
          >
            <Text style={{color: 'white', fontSize: 20}}>Train 2 epochs</Text>
          </TouchableOpacity>
        </View>
        <View style={{marginTop:10}}>
          <Button title='Reset model' color='#007AFFAF' onPress={() => resetModel().then(res => Alert.alert("Model reset to initial configuration"))}/>
        </View>
      </View>
  );
}

function round(n) {
  if (!n) {
    return 0.00.toFixed(2);
  }
  return (Math.floor(n * 100) / 100).toFixed(2);
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    paddingHorizontal: 10,
  },
  text: {
    textAlign: 'center',
    marginBottom: 10
  },
  buttonContainer: {
    flexDirection: 'row',
    alignItems: 'stretch',
    marginTop: 15,
    marginBottom: 20
  },
  button: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#eee',
    padding: 10,
  },
  middleButton: {
    borderLeftWidth: 1,
    borderRightWidth: 1,
    borderColor: '#ccc',
  },
});
