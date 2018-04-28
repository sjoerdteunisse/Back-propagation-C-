using static System.Console;

namespace Back_Propagation

{
    class Program
    {
        static void Main(string[] args)
        {
            //Create instance of our neural network
            NeuralNetwork neuralNetwork = new NeuralNetwork(new int[] { 3, 25, 25, 1 }, 0.333333f);

            for (int i = 0; i < 5000; i++)
            {
                //XOR Supervised
                neuralNetwork.FeedForward(new float[] { 0, 0, 0 });
                neuralNetwork.BackPropagation(new float[] { 0 });

                neuralNetwork.FeedForward(new float[] { 0, 0, 1 });
                neuralNetwork.BackPropagation(new float[] { 1 });

                neuralNetwork.FeedForward(new float[] { 0, 1, 0 });
                neuralNetwork.BackPropagation(new float[] { 1 });

                neuralNetwork.FeedForward(new float[] { 0, 1, 1 });
                neuralNetwork.BackPropagation(new float[] { 0 });

                neuralNetwork.FeedForward(new float[] { 1, 0, 0 });
                neuralNetwork.BackPropagation(new float[] { 1 });

                neuralNetwork.FeedForward(new float[] { 1, 0, 1 });
                neuralNetwork.BackPropagation(new float[] { 0 });

                neuralNetwork.FeedForward(new float[] { 1, 1, 0 });
                neuralNetwork.BackPropagation(new float[] { 0 });

                neuralNetwork.FeedForward(new float[] { 1, 1, 1 });
                neuralNetwork.BackPropagation(new float[] { 1 });
            }

            WriteLine(neuralNetwork.FeedForward(new float[] { 0, 0, 0 })[0]);
            WriteLine(neuralNetwork.FeedForward(new float[] { 0, 0, 1 })[0]);
            WriteLine(neuralNetwork.FeedForward(new float[] { 0, 1, 0 })[0]);
            WriteLine(neuralNetwork.FeedForward(new float[] { 0, 1, 1 })[0]);
            WriteLine(neuralNetwork.FeedForward(new float[] { 1, 1, 0 })[0]);
            WriteLine(neuralNetwork.FeedForward(new float[] { 1, 1, 1 })[0]);
            ReadKey();
        }
    }
}
