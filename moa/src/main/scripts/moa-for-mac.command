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
MAVEN_REPO="$(realpath ~)/.m2/repository"
JAR_PATHS="$(for j in $(find $MAVEN_REPO -name '*.jar');do printf '%s:' $j; done)"
CLASSPATH="$JAR_PATHS$REPO/"
JAVA_AGENT_PATH="$(find $MAVEN_REPO -name 'sizeofag-1.0.4.jar')"


JCMD=java
if [ -f "$(/usr/libexec/java_home -v 1.8.0_271)/bin/java" ]
then
#    JCMD="$(/usr/libexec/java_home -v 1.8.0_271)/bin/java -XX:+UnlockCommercialFeatures -XX:+FlightRecorder -XX:StartFlightRecording=delay=0s,duration=60s,name=myrecording,filename=/Users/ng98/Desktop/recordings/myrecording.jfr,settings=profile"
    JCMD="$(/usr/libexec/java_home -v 1.8.0_271)/bin/java"
fi

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
  $MAIN