#!/bin/bash

TOPDIR=`dirname $0`
TOPDIR=`realpath ${TOPDIR}`
BUILDDIR=/tmp/build/rpcio/build
#BUILDDIR=${TOPDIR}/build
mkdir -p ${BUILDDIR}
cd ${BUILDDIR}
cmake ${TOPDIR} && make -j48

