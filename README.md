### Operator framework 
Create your own operator framework to design the flow of your application.

`Operator` - the main entity that encapsulates similar operations together.
Eg: OnboardingOperator which has operations to understand and save user information like name, email, age, etc.

`Operation` - multiple operations reside within an operator. The operator calls these operations based on user inputs to do the desired operation.
Eg: setAge, setEmailAddress, setHeight.

The framework intelligently decides which operation to call and the required arguments from the user input.

### Steps:

1. Create your operator class. Examples in `test_onboarding.py` and `test_motivation.py`. Follow the docstring format for each function to specify the description of the operation and each parameter within the tool.
2. Train the `router` of your operator. The `router` decides what operations to call. Use script `train_onboarding_operator.py` as an example. Your router would be saved in the specified `output_folder` with the name of the operator.
3. Now run your operator class with the `router` path from step 2.

### Examples
Onboarding example

![onboarding.png](images%2Fonboarding.png)

Motivation example

![motivation.png](images%2Fmotivation.png)