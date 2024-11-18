Wandb Link: https://wandb.ai/jjanae39-western-governors-university/nyc_airbnb?nw=nwuserjjanae39


GitHub Link: https://github.com/JLJ55/Project-Build-an-ML-Pipeline-Starter.git

### Future Improvements for the Pipeline

As I reflect on the work done so far and look to the future, I see several areas where we can make this pipeline even better. Here are a few ideas I’m considering for upcoming releases:

1. **Making the Pipeline More Flexible**  
   Right now, the pipeline works great for the NYC Airbnb dataset, but I’d like to make it easier to adapt to other datasets. For instance, I could add configuration options to handle different data formats or custom columns without having to modify the code.

2. **Automating More Tests**  
   I’ve added some great tests for catching errors, like the geographical boundary checks, but there’s always room for improvement. In the future, I’d like to expand these tests to automatically flag things like unusual patterns in prices or invalid property types. This would make the pipeline more robust when handling unexpected data.

3. **Improving Performance**  
   While the pipeline performs well, I think there’s an opportunity to make it even faster. I could explore ways to parallelize certain steps, like data cleaning or model training, to reduce overall runtime. This would be especially useful as we start working with larger datasets.

4. **Adding More Metrics and Insights**  
   Right now, the pipeline reports metrics like mean absolute error and R² for the model. In future versions, I’d like to add more advanced metrics or even create visualizations that make it easier to understand the model’s strengths and weaknesses.

5. **Building a More User-Friendly Interface**  
   While I’m comfortable running the pipeline through command-line tools, I know this isn’t everyone’s favorite way to work. One idea I’m considering is adding a simple user interface—maybe a web dashboard—that would make it easier to configure and run the pipeline without touching the code.

6. **Expanding the Scope of the Pipeline**  
   The pipeline currently focuses on cleaning, training, and testing the model, but there’s potential to do more. For example, I could add a step for deploying the model to a live environment or integrating it with an API for real-time predictions.

7. **Exploring More Machine Learning Models**  
   Right now, the pipeline is built around a random forest model, which works well for this use case. However, I’d like to experiment with other algorithms, like gradient boosting or neural networks, to see if they can provide even better results.

8. **Streamlining Artifact Management**  
   I use Weights & Biases (W&B) for tracking artifacts, which has been incredibly helpful. But as the number of artifacts grows, I might look into better ways to organize and manage them, so it’s easier to find what I need later.
