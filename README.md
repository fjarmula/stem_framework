High level idea o the architecture of the framework:

1. The Genome (State): A nested Pydantic structure that defines everything the agent is at a given moment—its system
   persona, its decision-making logic,and its tool definitions.
2. The Differentiation Loop: A specialized LLM chain that performs "Environmental Analysis." It looks at the task class
   and determines the "ideal" phenotype for an agent in that niche.
3. The Regulatory Checkpoint (Safeguard): Before any change is applied to the Genome, a separate validation step must
   simulate or critique the change to ensure it doesn't break schema consistency or introduce logical loops.
4. Maturity Heuristics: A set of convergence checks. When the "Evolutionary Engine" stops suggesting significant
   changes (delta is low), the agent is considered "Mature."

When model is asked to find Fibonacci sequence and does not have the ability to run the code, this is indeterministic
behavior. The model will try to find a way to solve the problem, but it may not always be successful. The model may try
to use its existing knowledge to find a solution, or it may try to generate new code to solve the problem. However,
without the ability to run code, the model may not be able to verify that its solution is correct. This can lead to
errors or incorrect answers.
Or it may think at the training stage that the solution is correct but at the end it is found to be wrong