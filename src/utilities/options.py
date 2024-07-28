from dataclasses import dataclass, field
from datetime import datetime
from typing import Union


@dataclass
class LoggingOptions:
    """
     Options for configuring the Synthesis process.

   Logging:
       level (str): Logging level. Choices are DEBUG, INFO, WARNING, ERROR, CRITICAL. Default is INFO.
       file (str): Log file path. Default is logs/default.log.

   Note:
       When using a YAML configuration file, these options can be grouped under 'logging', 'synthesis_parameters', and 'solver' sections.
       For example:

       logging:
         level: "DEBUG"
       synthesis_parameters:
         max_iterations: 20
         operation_costs:
           '+': 1
           '-': 1
           '*': 3
           'ite': 4
           'neg': 2
       solver:
         name: "z3"
         timeout: 30000
   """
    level: str = field(
        default="INFO",
        metadata=dict(
            description="Logging level",
            type="str",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        ))
    file: str = field(
        default=f"logs/run_{datetime.now()}.log",
        metadata=dict(
            description="Log file path",
            type="str"
        ))


@dataclass
class SynthesisParameters:
    """
    Options for configuring the Synthesis process.

    Synthesis Parameters:
        strategy (str): Synthesis strategy. Choices are random_enumerative, fast_enumerative. Default is random_enumerative.
        candidate_generation (str): Candidate generation strategy. Choices are bottom_up, top_down, fast_enumerative. Default is bottom_up.
        max_iterations (int): Maximum number of iterations. Default is 500.
        max_depth (int): Maximum depth of generated expressions. Default is 5.
        max_complexity (int): Maximum complexity of generated expressions. Default is 5.
        random_seed (int): Random seed for all solvers. Default is 1234.
        randomise_each_iteration (bool): Randomise seed between each synthesis iteration. Default is False.
        max_candidates_at_each_depth (int): Maximum number of candidate programs to consider at each depth. Default is 10.
        min_const (int): Minimum constant to introduce into the candidate programs. Default is -2.
        max_const (int): Maximum constant to introduce into the candidate programs. Default is 2.
        operation_costs (Dict[str, int]): Key-Value pairs representing operation costs for adjustable weighting. 
                                          Default is {'+': 1, '-': 1, '*': 2, 'ite': 3, 'neg': 1}.
        initial_weight (float): Initial weight for operations in random candidate generation. Default is 1.0.
        weight_multiplier (float): Weight multiplier for operations in candidate generation. 
                                   1 = no diversity, >1 will bias towards previously chosen operations and <1 will bias away. Default is 1.0.
        custom_grammar (Optional[Dict[str, List[Union[str, Tuple]]]]): Custom grammar for candidate generation. Default is None.
        use_weighted_generator (bool): Use weighted top-down generator instead of regular top-down generator. Default is False.

    Note:
        When using a YAML configuration file, these options can be grouped under 'logging', 'synthesis_parameters', and 'solver' sections.
        For example:

        logging:
          level: "DEBUG"
        synthesis_parameters:
          max_iterations: 20
          operation_costs:
            '+': 1
            '-': 1
            '*': 3
            'ite': 4
            'neg': 2
          custom_grammar:
            S: ["T", ["+", "S", "S"], ["-", "S", "S"]]
            T: ["x", "y", "1", "2"]
          use_weighted_generator: true
        solver:
          name: "z3"
          timeout: 30000

        When providing a custom grammar, ensure it's in the correct format:
        - For non-weighted grammars: Each key in the dictionary should have a list of strings or tuples as its value.
        - For weighted grammars: Each key should have a list of tuples, where each tuple contains the production rule and its weight.

        The custom_grammar can also be provided via command-line as a JSON string:
        --synthesis_parameters__custom_grammar '{"S": ["T", ["+", "S", "S"], ["-", "S", "S"]], "T": ["x", "y", "1", "2"]}'

   """
    strategy: str = field(
        default="random_enumerative",
        metadata=dict(
            description="Synthesis strategy",
            type="str",
            choices=["random_enumerative", "fast_enumerative"]
        ))
    candidate_generation: str = field(
        default="bottom_up",
        metadata=dict(
            description="Candidate generation strategy",
            type="str",
            choices=["bottom_up", "top_down", "fast_enumerative"]
        ))
    max_iterations: int = field(
        default=500,
        metadata=dict(
            description="Maximum number of iterations",
            type="int"))
    max_depth: int = field(
        default=5,
        metadata=dict(
            description="Maximum depth of generated expressions",
            type="int"
        ))
    max_complexity: int = field(
        default=5,
        metadata=dict(
            description="Maximum complexity of generated expressions",
            type="int"
        ))
    random_seed: int = field(
        default=1234,
        metadata=dict(
            description="Random seed for all solvers",
            type="int"
        ))
    randomise_each_iteration: bool = field(
        default=False,
        metadata=dict(
            description="Randomise seed between each synthesis iteration",
            type="bool"
        ))
    max_candidates_at_each_depth: int = field(
        default=10,
        metadata=dict(
            description="Maximum number of candidate programs to consider at each depth",
            type="int"
        ))
    min_const: int = field(
        default=-2,
        metadata=dict(
            description="Minimum constant to introduce into the candidate programs",
            type="int"
        ))
    max_const: int = field(
        default=2,
        metadata=dict(
            description="Maximum constant to introduce into the candidate programs",
            type="int"
        ))
    operation_costs: dict[str, int] = field(
        default_factory=lambda: {'+': 1, '-': 1, '*': 2, 'ite': 3, 'neg': 1},
        metadata=dict(
            description="Operation costs for candidate generation",
            type="dict"
        )
    )
    initial_weight: float = field(
        default=1.0,
        metadata=dict(
            description="Initial weight for operations in random candidate generation",
            type="float"
        ))
    weight_multiplier: float = field(
        default=1.0,
        metadata=dict(
            description="Weight multiplier for operations in candidate generation. "
                        "1 = no diversity, >1 will bias towards previously chosen operations and <1 will bias away",
            type="float"
        ))
    custom_grammar: dict[str, list[Union[str, tuple]]] = field(
        default=None,
        metadata=dict(
            description="Custom grammar for candidate generation",
            type="dict[str, list[Union[str, tuple]]]"
        )
    )
    use_weighted_generator: bool = field(
        default=False,
        metadata=dict(
            description="Use weighted top-down generator instead of regular top-down generator",
            type="bool"
        )
    )


@dataclass
class SolverOptions:
    """
     Options for configuring the Synthesis process.

     Solver Options:
       name (str): SMT Solver to use. Choices are z3, cvc5 (experimental/alpha phase). Default is z3.
       timeout (int): SMT Solver Configuration - Timeout. Default is 30000.

     Note:
       When using a YAML configuration file, these options can be grouped under 'logging', 'synthesis_parameters', and 'solver' sections.
       For example:

       logging:
         level: "DEBUG"
       synthesis_parameters:
         max_iterations: 20
         operation_costs:
           '+': 1
           '-': 1
           '*': 3
           'ite': 4
           'neg': 2
       solver:
         name: "z3"
         timeout: 30000
     """

    name: str = field(
        default="z3",
        metadata=dict(
            description="SMT Solver to use",
            type="str",
            choices=["z3", "cvc5"]
        ))
    timeout: int = field(
        default=30000,
        metadata=dict(
            description="SMT Solver Configuration - Timeout",
            type="int"
        ))


@dataclass
class Options:
    """
   Options for configuring the Synthesis process.

   Other:
       input_source (str): Input source. Choices are STDIN or a path to a problem file.

   Note:
       When using a YAML configuration file, these options can be grouped under 'logging', 'synthesis_parameters', and 'solver' sections.
       For example:

       logging:
         level: "DEBUG"
       synthesis_parameters:
         max_iterations: 20
         operation_costs:
           '+': 1
           '-': 1
           '*': 3
           'ite': 4
           'neg': 2
       solver:
         name: "z3"
         timeout: 30000
   """
    logging: LoggingOptions = field(default_factory=LoggingOptions)
    synthesis_parameters: SynthesisParameters = field(default_factory=SynthesisParameters)
    solver: SolverOptions = field(default_factory=SolverOptions)
    input_source: str = field(
        default="stdin",
        metadata=dict(
            description="Source of the input problem (stdin or file path)",
            type="str"
        ))
