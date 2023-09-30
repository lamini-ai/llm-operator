## LLM Operator framework 
Create your own operator framework! 

Build an LLM framework to intelligently select different functions to perform in your application: from planning different operations to use (ie. functions, APIs, or tools to use) to invoking those operations.

For example, a user might say `who me? I am of age fifty nine, my friend.`. You want to extract the age from this message. So, you would expect the LLM to call a function `setAge` that extracts the correct age `{'age': 59}`.

#### Workflow: LLM Onboarding Operator
Here's an example. You are building an application with a chat-based onboarding flow that gathers information about the user's demographic information, e.g. age and height, as your LLM has a conversation with the user.

First create an `OnboardingOperator` by extending the `Operator` class:
```
class OnboardingOperator(Operator):
```

Create a `setAge` method inside the class. This is an example of an operation that the `OnboardingOperator` can invoke. For example, if a user sends a message like "I'm 36 years old", the `OnboardingOperator` can invoke the `setAge` method and extract the parameter `age` from the message.
```
def setAge(self, age: int):
```

To make this an understandable to an LLM like `OnboardingOperator`, a natural language description can be prompt-engineered to explain what it does, e.g. `set the age of a person`.
```
def setAge(self, age: int):
    """
    set the age of a person
    """
```

Next, adding prompt-engineered descriptions to the parameters also provides contextual information to the LLM Operator:
```
def setAge(self, age: int):
    """
    set the age of a person
    
    Parameters:
    age: age of the person in years.
    """
```

Finally, once the operator routes to this function, you can do whatever you want in this function with the extracted parameter `age`. For example, you can save the age to a database, or you can return a customer message with the age, or you can call another LLM.
```
def setAge(self, age: int):
    """
    set the age of a person
    
    Parameters:
    age: age of the person in years.
    """
    ### Do whatever you want here, e.g. save the age to database, do some analysis, etc.
    ### As a hello world example, this is returning a string with the extract age parameter
    return f"Hello! Your age has been set to {age}"
```

Now, add additional operations! In our example, you can see also setting the height of the user. 

Finally, you can then train the Operator to route to the right operations.


### Framework

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
Onboarding Operator example

```
User input: who me? I am of age fifty nine, my friend.
Selected operation: ['setAge']
Generated arguments: {'age': 59}
```

Motivation Operator example

```
User input: Schedule a workout for 10 pm today.
Selected operation: ['setReminder']
Generated arguments: {'workout_name': 'no-name', 'workout_time': '2023-03-10T22:00:00Z'}
```
