node {
	def image_id = "hlink:${env.BUILD_TAG}"
	stage("Checkout") {
		deleteDir()
		checkout scm
	}
	stage("Deploy") {
		host = "gp1.pop.umn.edu"
		deploy_target = "/pkg/ipums/programming/linking/hlink/deploys/${env.BRANCH_NAME}"      
		def target_exists = sh script: "ssh ${host} 'test -d ${deploy_target}'", returnStatus:true
		if (target_exists != 0) {
			sh "ssh ${host} 'mkdir ${deploy_target} && cd ${deploy_target} && mkdir scripts'"
			sh "ssh ${host} 'cd ${deploy_target} && git clone git@github.umn.edu:mpc/hlink.git'"
			sh "ssh ${host} 'cd ${deploy_target} && /pkg/ipums/programming/conda/v4.8/envs/hlink/bin/virtualenv -p 3.6.5 venv'"
		}				
		sh "ssh ${host} 'cd ${deploy_target}/hlink && git checkout ${env.BRANCH_NAME} && git pull origin ${env.BRANCH_NAME}'"
		sh "rsync -av ./deploy/hlink ${host}:${deploy_target}/scripts/hlink"
		sh "rsync -av ./deploy/global_conf.json ${host}:${deploy_target}/global_conf.json"
		sh "ssh ${host} 'cd ${deploy_target} && sed -i \'s/XXX_BRANCH/${env.BRANCH_NAME}/g\' global_conf.json && sed -i \'s/XXX_BRANCH/${env.BRANCH_NAME}/g\' scripts/hlink'"
		//sh "ssh ${host} 'cd ${deploy_target}/hlink/scala_jar && rm -rf target && /pkg/mpctools/bin/sbt assembly && cp ./target/scala-2.11/hlink_lib-assembly-1.0.jar ../hlink/spark/jars'"            
		sh "ssh ${host} 'cd ${deploy_target} && venv/bin/pip install ./hlink'"
  }

	/*stage("Build") {
		docker.build(image_id)
	}
	stage("Black") {
		sh "docker run ${image_id} black --check ."
	}
	stage("Flake8") {
		sh "docker run ${image_id} flake8 --count ."
	}
	stage("Test") {
		sh "docker run ${image_id} pytest hlink/tests/"
	}*/
}
