# Arguments -> File -> Dummy values
import sys
import time  # Import the time module to measure execution time

execution_timelog = []

class LogExecutionTime:  # Define a class for the decorator
    def __init__(self, func):  # Initialize the decorator with the function to be decorated
        self.func = func  # Store the function to be decorated

    def __get__(self, instance, owner):  # Define the descriptor method to handle instance methods
        return lambda *args, **kwargs: self(instance, *args, **kwargs)  # Return a lambda that passes the instance

    def __call__(self, *args, **kwargs):  # Make the class instance callable
        instance = args[0]  # Extract the instance from the arguments
        start_time = time.time()  # Record the start time
        result = self.func(instance, *args[1:], **kwargs)  # Call the original function with its arguments
        end_time = time.time()  # Record the end time
        execution_time = end_time - start_time  # Calculate the execution time
        #print(f"Execution time of {self.func.__name__}: {execution_time:.4f} seconds")  # Log the execution time
        execution_timelog.append(f"Execution time of {self.func.__name__}: {execution_time:.4f} seconds\n")
        return result  # Return the result of the original function call

def print_execution_timelog():
    for i in range(len(execution_timelog)):
        print(execution_timelog[i])

def save_execution_timelog(description = ["",], filename = "output.txt"):
    try:
        file = open(filename, "w")
        desc = ["Execution log\n"]
        if type(description) == dict:
            for key, value in description.items():
                desc.append(f"{key}:{value}\n")
        else:
            desc = description                
        file.writelines(desc)
        file.writelines(execution_timelog)
        file.close()
    except:
        pass

# Example usage:
'''
class ExampleClass:  # Define an example class to demonstrate the decorator
    @LogExecutionTime  # Apply the decorator to the method
    def example_method(self):  # Define a method in the class
        for _ in range(1000000):  # A sample computation to add some delay
            pass  # Do nothing

# Instantiate the example class and call the decorated method
example = ExampleClass()  # Create an instance of the ExampleClass
example.example_method()  # Call the decorated method to see the execution time log
'''

def get_external_input(dummy_values=[], from_file = "input.txt"):
    sz = len(dummy_values)
    f = sys.argv[1:]    
    g = dummy_values    
    
    if len(f) == 0:
        f = []        
        try:
            file = open(from_file)            
            if sz != 0:
                q = file.readline().split('#')[0]
                q.rstrip
                n = 0            
                while q != "" and (n < sz):
                    f.append(q)
                    q = file.readline().split('#')[0]
                    q.rstrip
                    n += 1    
            else:                
                q = file.readlines()
                f = list(map(str.rstrip, q))
            file.close()            
        except:
            f = []
            pass
    
    if sz == 0:
        return f
    
    for i in range(min(sz,len(f))):
        if type(dummy_values[i]) == type(f[i]):
            g[i] = f[i]
        elif type(dummy_values[i]) == int:
            try:
                g[i] = int(f[i])
            except:                
                pass
        elif type(dummy_values[i]) == float:
            try:
                g[i] = float(f[i])
            except:
                pass
    return g
    
#A = get_external_input([1,2,3,"hello",4,5])
#A = get_external_input()
#print(A)
    
        
    