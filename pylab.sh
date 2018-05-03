#!/usr/bin/env bash

P=$1

ssh -L ${P}888:rack-jonathan-g0${P}:8888 yotamfra@nova.cs.tau.ac.il