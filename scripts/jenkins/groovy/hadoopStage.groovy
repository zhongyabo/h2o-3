H2O_HADOOP_STARTUP_MODE_HADOOP='ON_HADOOP'
H2O_HADOOP_STARTUP_MODE_STANDALONE='STANDALONE'
H2O_HADOOP_STARTUP_MODE_WITH_KRB='WITH_KRB'

def call(final pipelineContext, final stageConfig) {
    withCredentials([usernamePassword(credentialsId: 'ldap-credentials', usernameVariable: 'LDAP_USERNAME', passwordVariable: 'LDAP_PASSWORD')]) {
        stageConfig.customBuildAction = """
            if [ -n "\$HADOOP_CONF_DIR" ]; then
                export HADOOP_CONF_DIR=\$(realpath \${HADOOP_CONF_DIR})
            fi

            source /usr/sbin/hive_version_check.sh

            echo "Activating Python ${stageConfig.pythonVersion}"
            . /envs/h2o_env_python${stageConfig.pythonVersion}/bin/activate

            echo 'Initializing Hadoop environment...'
            sudo -E /usr/sbin/startup.sh

            echo 'Generating SSL Certificate'
            rm -f mykeystore.jks
            keytool -genkey -dname "cn=Mr. Jenkins, ou=H2O-3, o=H2O.ai, c=US" -alias h2o -keystore mykeystore.jks -storepass h2oh2o -keypass h2oh2o -keyalg RSA -keysize 2048

            echo 'Starting H2O on Hadoop'
            ${getH2OStartupCmd(stageConfig)}
            if [ -z \${CLOUD_IP} ]; then
                echo "CLOUD_IP must be set"
                exit 1
            fi
            if [ -z \${CLOUD_PORT} ]; then
                echo "CLOUD_PORT must be set"
                exit 1
            fi
            echo "Cloud IP:PORT ----> \$CLOUD_IP:\$CLOUD_PORT"

            echo "Running Make"
            make -f ${pipelineContext.getBuildConfig().MAKEFILE_PATH} ${stageConfig.target}${getMakeTargetSuffix(stageConfig)} check-leaks
        """
        
        stageConfig.postFailedBuildAction = getPostFailedBuildAction(stageConfig.customData.mode)

        def defaultStage = load('h2o-3/scripts/jenkins/groovy/defaultStage.groovy')
        try {
            defaultStage(pipelineContext, stageConfig)
        } finally {
            sh "find ${stageConfig.stageDir} -name 'h2odriver*.jar' -type f -delete -print"
        }
    }
}

private GString getH2OStartupCmd_hadoop(final stageConfig) {
    return """
            rm -fv h2o_one_node h2odriver.out
            hadoop jar h2o-hadoop-*/h2o-${stageConfig.customData.distribution}${stageConfig.customData.version}-assembly/build/libs/h2odriver.jar \\
                -n 1 -mapperXmx 2g -baseport 54445 \\
                -jks mykeystore.jks \\
                -notify h2o_one_node -ea -proxy \\
                -jks mykeystore.jks \\
                -login_conf ${stageConfig.customData.ldapConfigPath} -ldap_login \\
                &> h2odriver.out &
            for i in \$(seq 20); do
              if [ -f 'h2o_one_node' ]; then
                echo "H2O started on \$(cat h2o_one_node)"
                break
              fi
              echo "Waiting for H2O to come up (\$i)..."
              sleep 3
            done
            if [ ! -f 'h2o_one_node' ]; then
              echo 'H2O failed to start!'
              cat h2odriver.out
              exit 1
            fi
            IFS=":" read CLOUD_IP CLOUD_PORT < h2o_one_node
            export CLOUD_IP=\$CLOUD_IP
            export CLOUD_PORT=\$CLOUD_PORT
        """
}

private GString getH2OStartupCmd_kerberos(final stageConfig) {
    def defaultPort = 54321
    return """
            java -Djavax.security.auth.useSubjectCredsOnly=false \\
                -cp build/h2o.jar:\$(cat /opt/hive-jdbc-cp) water.H2OApp \\
                -port ${defaultPort} -ip \$(hostname --ip-address) -name \$(date +%s) \\
                -jks mykeystore.jks \\
                -spnego_login -user_name ${stageConfig.customData.kerberosUserName} \\
                -login_conf ${stageConfig.customData.kerberosConfigPath} \\
                -spnego_properties ${stageConfig.customData.kerberosPropertiesPath} \\
                > standalone_h2o.log 2>&1 & sleep 15
            export KERB_PRINCIPAL=${stageConfig.customData.kerberosPrincipal}
            export CLOUD_IP=\$(hostname --ip-address)
            export CLOUD_PORT=${defaultPort}
        """
}

/**
 * Returns the cmd used to start H2O in given mode (on Hadoop or standalone). The cmd <strong>must</strong> export
 * the CLOUD_IP and CLOUT_PORT env variables (they are checked afterwards).
 * @param stageConfig stage configuration to read mode and additional information from
 * @return the cmd used to start H2O in given mode
 */
private GString getH2OStartupCmd(final stageConfig) {
    switch (stageConfig.customData.mode) {
        case H2O_HADOOP_STARTUP_MODE_HADOOP:
            return getH2OStartupCmd_hadoop(stageConfig)
        case H2O_HADOOP_STARTUP_MODE_STANDALONE:
            def defaultPort = 54321
            return """
                java -cp build/h2o.jar:\$(cat /opt/hive-jdbc-cp) water.H2OApp \\
                    -port ${defaultPort} -ip \$(hostname --ip-address) -name \$(date +%s) \\
                    -jks mykeystore.jks \\
                    > standalone_h2o.log 2>&1 & sleep 15
                export CLOUD_IP=\$(hostname --ip-address)
                export CLOUD_PORT=${defaultPort}
            """
        case H2O_HADOOP_STARTUP_MODE_WITH_KRB:
            return getH2OStartupCmd_kerberos(stageConfig)
        default:
            error("Startup mode ${stageConfig.customData.mode} for H2O with Hadoop is not supported")
    }
}

private String getMakeTargetSuffix(final stageConfig) {
    switch (stageConfig.customData.mode) {
        case H2O_HADOOP_STARTUP_MODE_HADOOP:
            return "-hdp"
        case H2O_HADOOP_STARTUP_MODE_STANDALONE:
            return "-ldap"
        case H2O_HADOOP_STARTUP_MODE_WITH_KRB:
            return "-kerb"
        default:
            error("Startup mode ${stageConfig.customData.mode} for H2O with Hadoop is not supported")
    }
}


private String getPostFailedBuildAction(final mode) {
    switch (mode) {
        case H2O_HADOOP_STARTUP_MODE_HADOOP:
            return """
                if [ -f h2o_one_node ]; then
                    export YARN_APPLICATION_ID=\$(cat h2o_one_node | grep job | sed 's/job/application/g')
                    echo "YARN Application ID is \${YARN_APPLICATION_ID}"
                    yarn application -kill \${YARN_APPLICATION_ID}
                    yarn logs -applicationId \${YARN_APPLICATION_ID} > h2o_yarn.log 
                fi   
            """
        default:
            return ""
    }
}

return this
