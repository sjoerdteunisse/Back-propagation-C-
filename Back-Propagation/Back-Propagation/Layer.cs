using System;

namespace Back_Propagation
{
    public class Layer
    {
        private int amountOfInputs;
        private int amountOfOutputs;
        private float learningRate;

        public float[] outputs;
        public float[] inputs;
        public float[,] weights;
        public float[,] weightsDelta;
        public float[] gamma;
        public float[] error;

        public static Random rnd = new Random();

        public Layer()
        {
        }

        public Layer(int amountOfInputs, int amountOfOutputs, float learningRate)
        {
            this.amountOfInputs = amountOfInputs;
            this.amountOfOutputs = amountOfOutputs;
            this.learningRate = learningRate;

            outputs = new float[amountOfOutputs];
            inputs = new float[amountOfInputs];
            weights = new float[amountOfInputs, amountOfOutputs];
            weightsDelta = new float[amountOfInputs, amountOfOutputs];
            gamma = new float[amountOfOutputs];
            gamma = new float[amountOfOutputs];

            InitializeWeights();
        }

        public void InitializeWeights()
        {
            for (int i = 0; i < amountOfOutputs; i++)
            {
                for (int j = 0; j < amountOfInputs; j++)
                {
                    weights[i, j] = (float)rnd.NextDouble() - 0.5f;
                }
            }
        }

        public float[] FeedForward(float[] inputs)
        {
            this.inputs = inputs;
            for (int i = 0; i < amountOfOutputs; i++)
            {
                outputs[i] = 0;

                for (int j = 0; j < amountOfInputs; j++)
                {
                    outputs[i] += inputs[j] * weights[i, j];
                }

                outputs[i] = (float)Math.Tanh(outputs[i]);
            }

            return outputs;
        }

        public float TanHDer(float value)
        {
            return 1 - (value * value);
        }

        public void BackPropagationOutput(float[] expected)
        {
            for (int i = 0; i < amountOfOutputs; i++)
                error[i] = outputs[i] - expected[i];

            for (int i = 0; i < amountOfOutputs; i++)
                gamma[i] = error[i] * TanHDer(outputs[i]);

            for (int i = 0; i < amountOfOutputs; i++)
            {
                for (int j = 0; j < amountOfInputs; j++)
                {
                    weightsDelta[i, j] = gamma[i] * inputs[j];
                }
            }
        }

        public void BackPropagationHidden(float[] gammaForward, float[,] weightsForward)
        {
            for (int i = 0; i < amountOfOutputs; i++)
            {
                gamma[i] = 0;

                for (int j = 0; j < gammaForward.Length; j++)
                {
                    gamma[i] += gammaForward[j] * weightsForward[j, i];
                }

                gamma[i] *= TanHDer(outputs[i]);
            }
        }

        public void UpdateWeights()
        {
            for (int i = 0; i < amountOfOutputs; i++)
            {
                for (int j = 0; j < amountOfInputs; j++)
                {
                    //w1,1 = w -= d1,1 * lr 
                    weights[i, j] -= weightsDelta[i, j] * learningRate;
                }
            }
        }
    }
}
