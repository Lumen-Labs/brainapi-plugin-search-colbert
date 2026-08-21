from src.core.plugins.context import PluginContext


def register(context: PluginContext):
    from index import retrieve

    context.register_search_retriever("colbert", retrieve)
    if context._app:
        from routes import create_router

        context.include_router(create_router(context))
