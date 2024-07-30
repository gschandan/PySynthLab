#!/usr/bin/env python3
# simple_decomposer.py ---
#
# Filename: simple_decomposer.py
# Author: Arjun Radhakrishna
# Created: Sun, 11 Jun 2017 18:29:00 -0400
#
#
# Copyright (c) 2017, Arjun Radhakrishna, University of Pennsylvania
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. All advertising materials mentioning features or use of this software
#    must display the following acknowledgement:
#    This product includes software developed by The University of Pennsylvania
# 4. Neither the name of the University of Pennsylvania nor the
#    names of its contributors may be used to endorse or promote products
#    derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#

from utils import basetypes
from exprs import exprs
from semantics import semantics_types
from utils import z3smt

class DecomposerInterface(object):
    def __init__(self, syn_ctx, synth_funs, spec):
        self.syn_ctx = syn_ctx
        self.synth_funs = synth_funs
        self.spec = spec

    def decompose(self):
        raise basetypes.AbstractMethodError("DecomposerInterface.decompose()")

class DeterministicDecomposer(DecomposerInterface):
    def __init__(self, syn_ctx, synth_funs, spec):
        super().__init__(syn_ctx, synth_funs, spec)
        if spec.is_multipoint:
            assert False 

    def is_deterministic(self):
        smt_ctx = z3smt.Z3SMTContext()
        spec_formula = self.spec.spec_expr
        apps = []
        for sf in self.synth_funs:
            sf_apps = set(exprs.find_all_applications(spec_formula, sf.function_name))
            assert len(sf_apps) == 1
            apps.append(sf_apps.pop())

        outvars = []
        outpvars = []
        for sf in self.synth_funs:
            outvars.append(exprs.VariableExpression(exprs.VariableInfo(sf.range_type, 'out_' + sf.function_name)))
            outpvars.append(exprs.VariableExpression(exprs.VariableInfo(sf.range_type, 'outp_' + sf.function_name)))

        spec1 = exprs.substitute_all(spec_formula, list(zip(apps,  outvars)))
        spec2 = exprs.substitute_all(spec_formula, list(zip(apps,  outpvars)))
        ineq_constr = self.syn_ctx.make_function_expr('or', 
                *list(map(lambda ov, ovp: self.syn_ctx.make_function_expr('ne', ov, ovp), outvars, outpvars)))
        full_constr = self.syn_ctx.make_function_expr('and', spec1, spec2, ineq_constr)

        point = exprs.sample(full_constr, smt_ctx, {})
        return (point is None)

    def separate(self):
        spec_expr = self.spec.spec_expr
        if exprs.is_application_of(spec_expr, 'and'):
            constraints = spec_expr.children
        else:
            constraints = [ spec_expr ]
        for c in constraints:
            print(exprs.expression_to_string(c))
        assert False

    def decompose(self):
        if len(self.synth_funs) == 1:
            return None
        if not self.is_deterministic():
            return None
        self.separate()

        assert False

        # Should do this for all subsets of constraints
        # Currently approximate with only subsets of size 1
        
