#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 13:14:42 2022

@author: frank
"""
import streamlit as st

header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
modelTraining = st.beta_container()

with header:
    st.title("A Simple Stress Detection Machine Learning Model from Text on Social Media")

