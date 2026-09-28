# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 12:20:37 2026

@author: Adam Thomasson
"""

# core/boundary.py

def neumann_left_1d(field, i, *args):
    """
    Left Neumann BC:
        phi_1 - phi_0 = 0
    """
    return field[i + 1] - field[i]


def neumann_right_1d(field, i, *args):
    """
    Right Neumann BC:
        phi_N - phi_{N-1} = 0
    """
    return field[i] - field[i - 1]


def dirichlet_zero(field, i, *args):
    """
    Zero Dirichlet / gauge-fixing row:
        phi_i = 0
    """
    return field[i]


def dirichlet_value(field, i, value, *args):
    """
    General Dirichlet BC:
        phi_i - value = 0
    """
    return field[i] - value


def periodic_match(field, i, partner_index, *args):
    """
    Generic periodic-identification row:
        phi_i - phi_partner = 0
    """
    return field[i] - field[partner_index]


def left_reflecting_1d(field, i, *args):
    """
    Alias for left Neumann on a symmetric boundary.
    """
    return field[i + 1] - field[i]


def right_reflecting_1d(field, i, *args):
    """
    Alias for right Neumann on a symmetric boundary.
    """
    return field[i] - field[i - 1]


def value_match(field, i, other_index, *args):
    """
    Generic equality constraint:
        phi_i - phi_other = 0
    """
    return field[i] - field[other_index]


def derivative_zero_central_1d(field, left_index, right_index, dx, *args):
    """
    Central first-derivative zero condition:
        (phi_right - phi_left) / (2 dx) = 0
    """
    return (field[right_index] - field[left_index]) / (2.0 * dx)


def second_derivative_zero_1d(field, left_index, centre_index, right_index, dx, *args):
    """
    Zero second-derivative condition:
        (phi_right - 2 phi_centre + phi_left) / dx^2 = 0
    """
    return (field[right_index] - 2.0 * field[centre_index] + field[left_index]) / (dx * dx)