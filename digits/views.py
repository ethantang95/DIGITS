# Copyright (c) 2014-2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import glob
import json
import platform
import traceback

import flask
from flask.ext.socketio import join_room, leave_room
from werkzeug import HTTP_STATUS_CODES
import werkzeug.exceptions

from .config import config_value
from .webapp import app, socketio, scheduler
import digits
from digits import dataset, model, utils
from digits.log import logger
from digits.utils.routing import request_wants_json

blueprint = flask.Blueprint(__name__, __name__)

@blueprint.route('/index.json', methods=['GET'])
@blueprint.route('/', methods=['GET'])
def home():
    """
    DIGITS home page
    Returns information about each job on the server

    Returns JSON when requested:
        {
            datasets: [{id, name, status},...],
            models: [{id, name, status},...]
        }
    """
    running_datasets    = get_job_list(dataset.DatasetJob, True)
    completed_datasets  = get_job_list(dataset.DatasetJob, False)
    running_models      = get_job_list(model.ModelJob, True)
    completed_models    = get_job_list(model.ModelJob, False)

    if request_wants_json():
        data = {
                'version': digits.__version__,
                'jobs_dir': config_value('jobs_dir'),
                'datasets': [j.json_dict()
                    for j in running_datasets + completed_datasets],
                'models': [j.json_dict()
                    for j in running_models + completed_models],
                }
        if config_value('server_name'):
            data['server_name'] = config_value('server_name')
        return flask.jsonify(data)
    else:
        new_dataset_options = {
                                'Images': {
                                    'image-classification': {
                                        'title': 'Classification',
                                        'url': flask.url_for('digits.dataset.images.classification.views.new'),
                                    },
                                    'image-generic': {
                                        'title': 'Other',
                                        'url': flask.url_for('digits.dataset.images.generic.views.new'),
                                    },
                                },
                            }

        new_model_options = {
                                'Images': {
                                    'image-classification': {
                                        'title': 'Classification',
                                        'url': flask.url_for('digits.model.images.classification.views.new'),
                                    },
                                    'image-generic': {
                                        'title': 'Other',
                                        'url': flask.url_for('digits.model.images.generic.views.new'),
                                    },
                                },
                            }

        return flask.render_template('home.html',
                new_dataset_options = new_dataset_options,
                running_datasets    = running_datasets,
                completed_datasets  = completed_datasets,
                new_model_options   = new_model_options,
                running_models      = running_models,
                completed_models    = completed_models,
                total_gpu_count     = len(scheduler.resources['gpus']),
                remaining_gpu_count = sum(r.remaining() for r in scheduler.resources['gpus']),
                )

@blueprint.route('/completed_jobs.json', methods=['GET'])
def completed_jobs():
    """
    Returns JSON
        {
            datasets: [{id, name, status, status_css, submitted, elapsed, badge}],
            models:   [{id, name, status, status_css, submitted, elapsed, badge}],
        }
    """
    completed_datasets  = get_job_list(dataset.DatasetJob, False)
    completed_models    = get_job_list(model.ModelJob, False)

    def json_dict(job):
        d = {
            'id': job.id(),
            'name': job.name(),
            'status': job.status_of_tasks().name,
            'status_css': job.status_of_tasks().css,
            'submitted': job.status_history[0][1],
            'elapsed': job.runtime_of_tasks(),
        }

        if 'train_db_task' in dir(job):
            d.update({
                'backend': job.train_db_task().backend,
            })

        if 'train_task' in dir(job):
            d.update({
                'framework': job.train_task().get_framework_id(),
            })

        if hasattr(job, 'dataset_id'):
            d.update({
                'dataset_id': job.dataset_id,
            })

        return d

    data = {
        'datasets': [ json_dict(j) for j in completed_datasets ],
        'models': [ json_dict(j) for j in completed_models ],
    }

    return flask.jsonify(data)

def get_job_list(cls, running):
    return sorted(
            [j for j in scheduler.jobs.values() if isinstance(j, cls) and j.status.is_running() == running],
            key=lambda j: j.status_history[0][1],
            reverse=True,
            )


### Authentication/login

@blueprint.route('/login', methods=['GET','POST'])
def login():
    """
    Ask for a username (no password required)
    Sets a cookie
    """
    # Get the URL to redirect to after logging in
    next_url = utils.routing.get_request_arg('next') or \
            flask.request.referrer or flask.url_for('.home')

    if flask.request.method == 'GET':
        return flask.render_template('login.html', next=next_url)

    # Validate username
    username = utils.routing.get_request_arg('username').strip()
    try:
        utils.auth.validate_username(username)
    except ValueError as e:
        # Invalid username
        flask.flash(e.message, 'danger')
        return flask.render_template('login.html', next=next_url)

    # Valid username
    response = flask.make_response(flask.redirect(next_url))
    response.set_cookie('username', username)
    return response

@blueprint.route('/logout', methods=['GET','POST'])
def logout():
    """
    Unset the username cookie
    """
    next_url = utils.routing.get_request_arg('next') or \
            flask.request.referrer or flask.url_for('.home')

    response = flask.make_response(flask.redirect(next_url))
    response.set_cookie('username', '', expires=0)
    return response


### Jobs routes

@blueprint.route('/jobs/<job_id>', methods=['GET'])
def show_job(job_id):
    """
    Redirects to the appropriate /datasets/ or /models/ page
    """
    job = scheduler.get_job(job_id)
    if job is None:
        raise werkzeug.exceptions.NotFound('Job not found')

    if isinstance(job, dataset.DatasetJob):
        return flask.redirect(flask.url_for('digits.dataset.views.show', job_id=job_id))
    if isinstance(job, model.ModelJob):
        return flask.redirect(flask.url_for('digits.model.views.show', job_id=job_id))
    else:
        raise werkzeug.exceptions.BadRequest('Invalid job type')

@blueprint.route('/jobs/<job_id>', methods=['PUT'])
@utils.auth.requires_login(redirect=False)
def edit_job(job_id):
    """
    Edit a job's name and/or notes
    """
    job = scheduler.get_job(job_id)
    if job is None:
        raise werkzeug.exceptions.NotFound('Job not found')

    if not utils.auth.has_permission(job, 'edit'):
        raise werkzeug.exceptions.Forbidden()

    # Edit name
    if 'job_name' in flask.request.form:
        name = flask.request.form['job_name'].strip()
        if not name:
            raise werkzeug.exceptions.BadRequest('name cannot be blank')
        job._name = name
        # update form data so updated name gets used when cloning job
        if 'form.dataset_name.data' in job.form_data:
            job.form_data['form.dataset_name.data'] = name
        elif 'form.model_name.data' in job.form_data:
            job.form_data['form.model_name.data'] = name
        else:
            # we are utterly confused
            raise werkzeug.exceptions.BadRequest('Unable to edit job type %s' % job.job_type())
        logger.info('Set name to "%s".' % job.name(), job_id=job.id())

    # Edit notes
    if 'job_notes' in flask.request.form:
        notes = flask.request.form['job_notes'].strip()
        if not notes:
            notes = None
        job._notes = notes
        logger.info('Updated notes.', job_id=job.id())

    return '%s updated.' % job.job_type()

@blueprint.route('/datasets/<job_id>/status', methods=['GET'])
@blueprint.route('/models/<job_id>/status', methods=['GET'])
@blueprint.route('/jobs/<job_id>/status', methods=['GET'])
def job_status(job_id):
    """
    Returns a JSON objecting representing the status of a job
    """
    job = scheduler.get_job(job_id)
    result = {}
    if job is None:
        result['error'] = 'Job not found.'
    else:
        result['error'] = None
        result['status'] = job.status.name
        result['name'] = job.name()
        result['type'] = job.job_type()
    return json.dumps(result)

@blueprint.route('/datasets/<job_id>', methods=['DELETE'])
@blueprint.route('/models/<job_id>', methods=['DELETE'])
@blueprint.route('/jobs/<job_id>', methods=['DELETE'])
@utils.auth.requires_login(redirect=False)
def delete_job(job_id):
    """
    Deletes a job
    """
    job = scheduler.get_job(job_id)
    if job is None:
        raise werkzeug.exceptions.NotFound('Job not found')

    if not utils.auth.has_permission(job, 'delete'):
        raise werkzeug.exceptions.Forbidden()

    try:
        if scheduler.delete_job(job_id):
            return 'Job deleted.'
        else:
            raise werkzeug.exceptions.Forbidden('Job not deleted')
    except utils.errors.DeleteError as e:
        raise werkzeug.exceptions.Forbidden(str(e))

@blueprint.route('/jobs', methods=['DELETE'])
@utils.auth.requires_login(redirect=False)
def delete_jobs():
    """
    Deletes a list of jobs
    """
    not_found = 0
    forbidden = 0
    failed = 0
    job_ids = flask.request.form.getlist('job_ids[]')
    for job_id in job_ids:

        try:
            job = scheduler.get_job(job_id)
            if job is None:
                print '%s not found' % job_id
                not_found += 1
                continue

            if not utils.auth.has_permission(job, 'delete'):
                print 'delete %s forbidden' % job_id
                forbidden += 1
                continue

            if not scheduler.delete_job(job_id):
                print 'delete %s failed' % job_id
                failed += 1
                continue
        except Exception as e:
            error.append(e)
            pass

    error = []
    if not_found:
        error.append('%d job%s not found.' % (not_found, '' if not_found == 1 else 's'))

    if forbidden:
        error.append('%d job%s not permitted to be deleted.' % (forbidden, '' if forbidden == 1 else 's'))

    if failed:
        error.append('%d job%s failed to delete.' % (failed, '' if failed == 1 else 's'))

    if len(error) > 0:
        error = ' '.join(error)
        raise werkzeug.exceptions.BadRequest(error)

    return 'Jobs deleted.'

@blueprint.route('/datasets/<job_id>/abort', methods=['POST'])
@blueprint.route('/models/<job_id>/abort', methods=['POST'])
@blueprint.route('/jobs/<job_id>/abort', methods=['POST'])
@utils.auth.requires_login(redirect=False)
def abort_job(job_id):
    """
    Aborts a running job
    """
    job = scheduler.get_job(job_id)
    if job is None:
        raise werkzeug.exceptions.NotFound('Job not found')

    if scheduler.abort_job(job_id):
        return 'Job aborted.'
    else:
        raise werkzeug.exceptions.Forbidden('Job not aborted')

@blueprint.route('/clone/<clone>', methods=['POST', 'GET'])
@utils.auth.requires_login
def clone_job(clone):
    """
    Clones a job with the id <clone>, populating the creation page with data saved in <clone>
    """

    ## <clone> is the job_id to clone

    job = scheduler.get_job(clone)
    if job is None:
        raise werkzeug.exceptions.NotFound('Job not found')

    if isinstance(job, dataset.ImageClassificationDatasetJob):
        return flask.redirect(flask.url_for('digits.dataset.images.classification.views.new') + '?clone=' + clone)
    if isinstance(job, dataset.GenericImageDatasetJob):
        return flask.redirect(flask.url_for('digits.dataset.images.generic.views.new') + '?clone=' + clone)
    if isinstance(job, model.ImageClassificationModelJob):
        return flask.redirect(flask.url_for('digits.model.images.classification.views.new') + '?clone=' + clone)
    if isinstance(job, model.GenericImageModelJob):
        return flask.redirect(flask.url_for('digits.model.images.generic.views.new') + '?clone=' + clone)
    else:
        raise werkzeug.exceptions.BadRequest('Invalid job type')

### Error handling

@app.errorhandler(Exception)
def handle_error(e):
    """
    Handle errors, formatting them as JSON if requested
    """
    error_type = type(e).__name__
    message = str(e)
    trace = None
    description = None
    status_code = 500
    if isinstance(e, werkzeug.exceptions.HTTPException):
        status_code = e.code
        description = e.description
    if app.debug:
        trace = traceback.format_exc()

    if request_wants_json():
        details = {
                'message': message,
                'type': error_type,
                }
        if description is not None:
            details['description'] = description
        if trace is not None:
            details['trace'] = trace.split('\n')
        return flask.jsonify({'error': details}), status_code
    else:
        return flask.render_template('error.html',
                title       = error_type,
                message     = message,
                description = description,
                trace       = trace,
                ), status_code

# Register this handler for all error codes
# Necessary for flask<=0.10.1
for code in HTTP_STATUS_CODES:
    if code not in [301]:
        app.register_error_handler(code, handle_error)

### File serving

@blueprint.route('/files/<path:path>', methods=['GET'])
def serve_file(path):
    """
    Return a file in the jobs directory

    If you install the nginx.site file, nginx will serve files instead
    and this path will never be used
    """
    jobs_dir = config_value('jobs_dir')
    return flask.send_from_directory(jobs_dir, path)

### Path Completion

@blueprint.route('/autocomplete/path', methods=['GET'])
def path_autocomplete():
    """
    Return a list of paths matching the specified preamble

    """
    path = flask.request.args.get('query','')
    suggestions = glob.glob(path+"*")
    if platform.system() == 'Windows':
        # on windows, convert backslashes with forward slashes
        suggestions = [p.replace('\\', '/') for p in suggestions]

    result = {
        "suggestions": sorted(suggestions)
    }

    return json.dumps(result)

### SocketIO functions

## /home

@socketio.on('connect', namespace='/home')
def on_connect_home():
    """
    Somebody connected to the homepage
    """
    pass

@socketio.on('disconnect', namespace='/home')
def on_disconnect_home():
    """
    Somebody disconnected from the homepage
    """
    pass

## /jobs

@socketio.on('connect', namespace='/jobs')
def on_connect_jobs():
    """
    Somebody connected to a jobs page
    """
    pass

@socketio.on('disconnect', namespace='/jobs')
def on_disconnect_jobs():
    """
    Somebody disconnected from a jobs page
    """
    pass

@socketio.on('join', namespace='/jobs')
def on_join_jobs(data):
    """
    Somebody joined a room
    """
    room = data['room']
    join_room(room)
    flask.session['room'] = room

@socketio.on('leave', namespace='/jobs')
def on_leave_jobs():
    """
    Somebody left a room
    """
    if 'room' in flask.session:
        room = flask.session['room']
        del flask.session['room']
        #print '>>> Somebody left room %s' % room
        leave_room(room)

