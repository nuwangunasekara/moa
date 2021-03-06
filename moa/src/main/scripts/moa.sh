#!/bin/bash
# ----------------------------------------------------------------------------
#  Copyright 2001-2006 The Apache Software Foundation.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ----------------------------------------------------------------------------

#   Copyright (c) 2001-2002 The Apache Software Foundation.  All rights
#   reserved.

#   Copyright (C) 2011-2019 University of Waikato, Hamilton, NZ

BASEDIR=`dirname $0`/..
BASEDIR=`(cd "$BASEDIR"; pwd)`
REPO=$BASEDIR/../../target/classes

if [ $# -gt 0 ]; then
  if [ -d "$1" ]; then
    export DJL_CACHE_DIR=$1
  else
    echo "DJL_CACHE_DIR can not be set. Directory $1 is not available."
    print_usage
    exit 1
  fi
fi

MAVEN_REPO="$(realpath ~)/.m2/repository"
if [ $# -gt 1 ]; then
  if [ -d "$2" ]; then
    MAVEN_REPO="$2"
    export MAVEN_OPTS="-Dmaven.repo.local=$2"
  else
    echo "MAVEN_OPTS=-Dmaven.repo.local can not be set. Directory $2 is not available."
    print_usage
    exit 1
  fi
fi


JAR_PATHS="$(for j in $(find $MAVEN_REPO -name '*.jar');do printf '%s:' $j; done)"
CLASSPATH="$JAR_PATHS$REPO/"
JAVA_AGENT_PATH="$(find $MAVEN_REPO -name 'sizeofag-1.0.4.jar')"


JCMD=java
if [ -f "$JAVA_HOME/bin/java" ]
then
  JCMD="$JAVA_HOME/bin/java"
fi

$JCMD -version

# check options
MEMORY=512m
MAIN=moa.gui.GUI
ARGS=
OPTION=
WHITESPACE="[[:space:]]"
for ARG in "$@"
do
  if [ "$ARG" = "-h" ] || [ "$ARG" = "-help" ] || [ "$ARG" = "--help" ]
  then
  	echo "Start script for MOA: Massive Online Analysis"
  	echo ""
  	echo "-h/-help/--help"
  	echo "    prints this help"
  	echo "-memory <memory>"
  	echo "    for supplying maximum heap size, eg 512m or 1g (default: 512m)"
  	echo "-main <classname>"
  	echo "    the class to execute (default: moa.gui.GUI)"
  	echo ""
  	echo "Note: any other options are passed to the Java class as arguments"
  	echo ""
  	exit 0
  fi

  if [ "$ARG" = "-memory" ] || [ "$ARG" = "-main" ]
  then
  	OPTION=$ARG
  	continue
  fi

  if [ "$OPTION" = "-memory" ]
  then
    MEMORY=$ARG
    OPTION=""
    continue
  elif [ "$OPTION" = "-main" ]
  then
    MAIN=$ARG
    OPTION=""
    continue
  fi

  if [[ $ARG =~ $WHITESPACE ]]
  then
    ARGS="$ARGS \"$ARG\""
  else
    ARGS="$ARGS $ARG"
  fi
done

# launch class
"$JCMD" \
  -classpath "$CLASSPATH" \
  -Xmx8g -Xms50m -Xss1g \
  -javaagent:"$JAVA_AGENT_PATH" \
  $MAIN \
  $ARGS
