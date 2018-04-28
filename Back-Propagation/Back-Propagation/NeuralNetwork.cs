namespace Back_Propagation
{
    public class NeuralNetwork
    {
        private int[] _layer;
        private Layer[] _layers;

        public NeuralNetwork()
        {
        }
         
        public NeuralNetwork(int[] layer, float learningRate)
        {
            _layer = new int[layer.Length];

            for (int i = 0; i < layer.Length; i++)
            {
                _layer[i] = layer[i];
            }

            _layers = new Layer[layer.Length - 1];

            for (int i = 0; i < _layers.Length; i++)
            {
                _layers[i] = new Layer(layer[i], layer[i + 1], learningRate);
            }
        }

        public float[] FeedForward(float[] inputs)
        {
            _layers[0].FeedForward(inputs);

            for (int i = 1; i < _layers.Length; i++)
            {
                _layers[i].FeedForward(_layers[i - 1].outputs);
            }

            return _layers[_layers.Length - 1].outputs;
        }

        public void BackPropagation(float[] expected)
        {
            for (int i = _layers.Length - 1; i >= 0; i--)
            {
                if (i == _layers.Length - 1)
                {
                    _layers[i].BackPropagationOutput(expected);
                }
                else
                {
                    _layers[i].BackPropagationHidden(_layers[i + 1].gamma, _layers[i + 1].weights);
                }
            }

            for (int i = 0; i < _layers.Length; i++)
            {
                _layers[i].UpdateWeights();
            }
        }
    }
}
