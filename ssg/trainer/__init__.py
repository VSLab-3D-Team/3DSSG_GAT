#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 18:44:20 2021

@author: sc
"""
from .trainer_SGFN import Trainer_SGFN
from .trainer_IMP import Trainer_IMP
from .trainer_MMAN import Trainer_MMAN
from .trainer_SGFN import Trainer_ALIGN
trainer_dict = {
    'sgfn': Trainer_SGFN,
    # 'sgpn': Trainer_SGFN,
    'sgpn': Trainer_ALIGN,
    'imp': Trainer_IMP,
    'jointsg': Trainer_SGFN,
    'mman': trainer_MMAN,
}
