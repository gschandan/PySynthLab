Main Module
===========

.. automodule:: src.runner
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Error Codes
-----------

The following error codes are used in the main module:

- 0: Success - A solution was found successfully.
- 1: General Error - An exception occurred during the synthesis process.
- 2: No Solution - The synthesis process completed, but no satisfying candidates were found.
- -6: Timeout - The synthesis process did not complete within the specified timeout period.

These error codes were chosen to be comparable with cvc5 solver's error codes for easier comparison.

Global Variables
----------------

.. py:data:: problem_global

   A global variable of type Optional[SynthesisProblemZ3 | SynthesisProblemCVC5] used to store the problem instance for metric logging.

Example Usage
-------------

To run the synthesis process:

.. code-block:: python

   if __name__ == "__main__":
       try:
           main()
       except Exception as e:
           print(f"An error occurred: {str(e)}")
           sys.exit(1)

This will execute the main function, which sets up the synthesis configuration, runs the synthesis process, and returns the results or any errors that occur.