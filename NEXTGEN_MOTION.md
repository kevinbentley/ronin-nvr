# Next Generation Motion and Object Detection Design

## Goal

The current approach of frame subtraction and yolo8l has unacceptable reliability. Motion and objects are missed, and false positives are frequent.

The goal of this effort is to use an advanced, next generation approach, to dramatically improve quality while maintaining performance capable of running 16 cameras on an 8-core CPU with 64GB RAM and dual NVIDIA 2070 GPUs.

## Research

The document docs/research/Motion_and_Object_Detection_Research.pdf contains the result of extensive research and analysis. This effort aims to implement the guidance and recommendations found in that document.

## Prototype, test, and development

A great deal of sample video data from real recordings can be found under the /opt3/ronin/storage directory.

In order to automate development, it is desireable to run Claude Code with --permission-mode bypassPermissions. To do this safely, we will create a docker for development. Once the docker is created, the user will install Claude Code and log in to it. At that point, claude will takeover inside the docker. The docker will have the /opt3/ronin/storage volume mounted, as well as the existing repository already checked out and mapped as a volume to the docker.

The docker will have access to the dual NVIDIA GPUs, and the prototype/test code should use it as if it were a production environment. The existing videos can be used to test the pipeline's accuracy and speed.

