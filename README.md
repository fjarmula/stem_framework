### High level idea o the architecture of the framework:

1. The Genome (State): A nested Pydantic structure that defines everything the agent is at a given moment—its system
   persona, its decision-making logic,and its tool definitions.
2. The Differentiation Loop: A specialized LLM chain that performs "Environmental Analysis." It looks at the task class
   and determines the "ideal" phenotype for an agent in that niche.
3. The Regulatory Checkpoint (Safeguard): Before any change is applied to the Genome, a separate validation step must
   simulate or critique the change to ensure it doesn't break schema consistency or introduce logical loops.
4. Maturity Heuristics: A set of convergence checks. When the "Evolutionary Engine" stops suggesting significant
   changes (delta is low), the agent is considered "Mature."

### Observations:

When model is asked to find Fibonacci sequence and does not have the ability to run the code, this is indeterministic
behavior. The model will try to find a way to solve the problem, but it may not always be successful. The model may try
to use its existing knowledge to find a solution, or it may try to generate new code to solve the problem. However,
without the ability to run code, the model may not be able to verify that its solution is correct. This can lead to
errors or incorrect answers.
Or it may think at the training stage that the solution is correct but at the end it is found to be wrong

### Future work:

As for now the agent learns how to use given tools, but it does not learn how to create new tools. The agent can only
use the tools that are defined in its Genome. It cannot create new tools on its own. However, the agent can learn to use
the existing tools more effectively over time through the differentiation loop and regulatory checkpoint processes.
The next step would be to implement a mechanism for the agent to propose new tools based on its interactions and
experiences. This could involve the agent identifying gaps in its capabilities and suggesting new tools that could fill
those gaps. The regulatory checkpoint would then evaluate these proposed tools for feasibility and safety before they
are added to the Genome.

Jude the agent's results based on unit tests. The agent can be given a set of unit tests that it must pass in order to
be considered successful. This would provide a more objective measure of the agent's performance and help ensure that it
is learning effectively. The agent could also be given feedback on its performance on these unit tests, which could help
guide its learning process and improve its performance over time.

Add inference with trained agents. Once the agent has gone through the differentiation loop and regulatory checkpoint
processes, it can be used for inference. This would involve using the trained agent to perform tasks or make decisions
based on its learned capabilities. The agent's performance during inference could be evaluated against a set of
benchmarks or real-world scenarios to assess its effectiveness and identify areas for further improvement.