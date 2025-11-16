
class FlaskWrapper(object):

    def __init__(self, app, **configs):
        self.app = app
        self.configs(**configs)

    def configs(self, **configs):
        for configure,value in configs:
            self.app.config[configure.upper()] = value

    def addEnpoint(self, endpoint=None, endpointName=None, handler=None, methods=['GET'], *args, **kwargs):
        self.app.add_url_rule(endpoint,endpointName,handler,methods=methods, *args, **kwargs)
    
    def registerBlueprint(self, bluePrint, **options):
        self.app.register_blueprint(bluePrint,**options)

    def run(self, **kwargs):
        self.app.run(**kwargs)

