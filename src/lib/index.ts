type Layer = Neuron[];

function ReLU(x: number) {
  return x < 0 ? 0 : x;
}

function ReLUDeriv(x: number) {
  return x > 0 ? 1 : 0;
}

function Sigmoid(x: number) {
  return 1 / (1 + Math.exp(-x));
}

function SigmoidDeriv(x: number) {
  return x * (1 - x);
}

function softmax(x: number[]) {
  const expValues = x.map((v) => Math.exp(v));
  const sum = expValues.reduce((acc, val) => acc + val, 0);
  return expValues.map((v) => v / sum);
}

function softmaxDeriv(softmaxOutput: number[], targetIndex: number) {
  return softmaxOutput.map((output, i) => {
    if (i === targetIndex) {
      return output * (1 - output);
    } else {
      return -output * softmaxOutput[targetIndex];
    }
  });
}

function computeNetwork(Network: Layer[], input: number[]) {
  const activations: number[][] = [input]; 
  const zValues: number[][] = []; 
  let currentInput = input;
  
  for (let i = 0; i < Network.length; i++) {
    const layerResult = computeLayerWithDetails(Network[i], currentInput, i === Network.length - 1);
    zValues.push(layerResult.zValues);
    currentInput = layerResult.activations;
    activations.push(currentInput);
  }
  
  const softmaxOutput = softmax(currentInput);
  activations[activations.length - 1] = softmaxOutput;
  
  return {
    output: softmaxOutput,
    activations: activations,
    zValues: zValues
  };
}

function computeLayerWithDetails(Layer: Layer, input: number[], isLastLayer = false) {
  const zValues: number[] = [];
  const activations: number[] = [];
  
  for (let i = 0; i < Layer.length; i++) {
    const z = Layer[i].compute(input);
    zValues.push(z);
    
    const activation = isLastLayer ? z : Sigmoid(z);
    activations.push(activation);
  }
  
  return { zValues, activations };
}

function crossEntropyLoss(output: number[], expected: number[]) {
  const loss = output.map((v, i) => -expected[i] * Math.log(v + 1e-10));
  return loss.reduce((acc, val) => acc + val, 0) / output.length;
}

class Neuron {
  bias: number;
  inputs: number[];
  weights: number[];
  lastInput: number[] = [];
  lastZ: number = 0;

  constructor(props?: { bias?: number; input?: number[]; weights?: number[] }) {
    this.bias = props?.bias ?? Math.random() * 2 - 1;
    this.inputs = props?.input ?? [];
    this.weights = props?.weights ?? [];
    this.lastInput = this.inputs;
    this.lastZ = this.bias;
  }

  compute = (input: number[]) => {
    const filteredInput =
      this.inputs.length === 0 ? input : input.filter((_, i) => this.inputs[i]);

    if (filteredInput.length !== this.weights.length) {
      this.weights = Array<number>(filteredInput.length).fill(Math.random() * 2 - 1);
    }

    this.lastInput = filteredInput;

    let res = 0;
    for (let i = 0; i < filteredInput.length; i++) {
      res += filteredInput[i] * this.weights[i];
    }

    this.lastZ = res + this.bias;
    return this.lastZ;
  };
}

function BuildNetwork(layers: number[]) {
  const Network: Layer[] = [];
  for (let i = 0; i < layers.length; i++) {
    const layer: Layer = [];
    for (let j = 0; j < layers[i]; j++) {
      layer.push(new Neuron());
    }
    Network.push(layer);
  }
  return Network;
}

function BackPropagate(Network: Layer[], input: number[], expected: number[], learningRate: number) {
  const forwardResult = computeNetwork(Network, input);
  const output = forwardResult.output;
  const activations = forwardResult.activations;
  
  const loss = crossEntropyLoss(output, expected);
  
  const outputLayerIndex = Network.length - 1;
  const outputLayerDelta: number[] = [];
  
  for (let i = 0; i < Network[outputLayerIndex].length; i++) {
    outputLayerDelta.push(output[i] - expected[i]);
  }
  
  const deltas: number[][] = new Array(Network.length);
  deltas[outputLayerIndex] = outputLayerDelta;
  
  for (let l = outputLayerIndex - 1; l >= 0; l--) {
    const layerDelta: number[] = [];
    
    for (let i = 0; i < Network[l].length; i++) {
      let error = 0;
      
      for (let j = 0; j < Network[l + 1].length; j++) {
        error += deltas[l + 1][j] * Network[l + 1][j].weights[i];
      }
      
      const sigmoidOutput = activations[l + 1][i];
      error *= SigmoidDeriv(sigmoidOutput);
      
      layerDelta.push(error);
    }
    
    deltas[l] = layerDelta;
  }
  
  for (let l = 0; l < Network.length; l++) {
    const layer = Network[l];
    const layerInputs = activations[l];
    
    for (let i = 0; i < layer.length; i++) {
      const neuron = layer[i];
      
      for (let j = 0; j < neuron.weights.length; j++) {
        const weightGradient = deltas[l][i] * layerInputs[j];
        neuron.weights[j] -= learningRate * weightGradient;
      }
      
      neuron.bias -= learningRate * deltas[l][i];
    }
  }
  
  return loss;
}

let Network: Layer[] = BuildNetwork([4, 5, 5, 10]);

const input = Array(4).fill(0).map(() => Math.random());
const expected = Array(10).fill(0).map(() => Math.random());
const sum = expected.reduce((acc, val) => acc + val, 0);
const normalizedExpected = expected.map((val) => val / sum);

const LearnRate = 0.0001;
const numIterations = 1000000;

let lastlost = 0
for (let i = 0; i < numIterations; i++) {
  const loss = BackPropagate(Network, input, normalizedExpected, LearnRate);
  const gain = Math.abs(loss - lastlost);

  if (gain < .000000000001) {
    console.log(`Iteration ${i + 1}: Gain = ${gain}, Loss = ${loss}`);
    console.log(`Converged at iteration ${i + 1}`);
    break;
  }

  if (i % 10000 === 0 || i === numIterations - 1) {
    console.log(`Iteration ${i + 1}: Gain = ${gain}, Loss = ${loss}`);
  }
  lastlost = loss;
}
