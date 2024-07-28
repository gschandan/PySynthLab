Candidate Generators
====================

.. automodule:: src.cegis.z3.candidate_generator
   :undoc-members:
   :no-index:

CandidateGenerator Base Class
-----------------------------

.. autoclass:: src.cegis.z3.candidate_generator.candidate_generator_base.CandidateGenerator
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

   .. automethod:: __init__
   .. automethod:: generate_candidates
   .. automethod:: prune_candidates
   .. automethod:: op_complexity
   .. automethod:: get_arg_sorts
   .. automethod:: create_candidate_function

TopDownCandidateGenerator Class
-------------------------------

.. autoclass:: src.cegis.z3.candidate_generator.top_down_enumerative_generator.TopDownCandidateGenerator
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

WeightedTopDownCandidateGenerator Class
---------------------------------------

.. autoclass:: src.cegis.z3.candidate_generator.weighted_top_down_enumerative_generator.WeightedTopDownCandidateGenerator
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

RandomCandidateGenerator Class
------------------------------

.. autoclass:: src.cegis.z3.candidate_generator.random_candidate_generator.RandomCandidateGenerator
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

FastEnumerativeCandidateGenerator Class
---------------------------------------

.. autoclass:: src.cegis.z3.candidate_generator.fast_enumerative_candidate_generator.FastEnumerativeCandidateGenerator
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:
