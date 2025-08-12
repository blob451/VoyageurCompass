# backend-tests
## Run Django check stage:

Run python manage.py check
Traceback (most recent call last):
  File "/home/runner/work/VoyageurCompass/VoyageurCompass/manage.py", line 22, in <module>
    main()
  File "/home/runner/work/VoyageurCompass/VoyageurCompass/manage.py", line 18, in main
    execute_from_command_line(sys.argv)
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/django/core/management/__init__.py", line 442, in execute_from_command_line
    utility.execute()
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/django/core/management/__init__.py", line 436, in execute
    self.fetch_command(subcommand).run_from_argv(self.argv)
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/django/core/management/base.py", line 408, in run_from_argv
    parser = self.create_parser(argv[0], argv[1])
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/django/core/management/base.py", line 371, in create_parser
    self.add_arguments(parser)
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/django/core/management/commands/check.py", line 47, in add_arguments
    choices=tuple(connections),
            ^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/django/utils/connection.py", line 73, in __iter__
    return iter(self.settings)
                ^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/django/utils/functional.py", line 47, in __get__
    res = instance.__dict__[self.name] = self.func(instance)
                                         ^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/django/utils/connection.py", line 45, in settings
    self._settings = self.configure_settings(self._settings)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/django/db/utils.py", line 148, in configure_settings
    databases = super().configure_settings(databases)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/django/utils/connection.py", line 50, in configure_settings
    settings = getattr(django_settings, self.settings_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/django/conf/__init__.py", line 81, in __getattr__
    self._setup(name)
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/django/conf/__init__.py", line 68, in _setup
    self._wrapped = Settings(settings_module)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/django/conf/__init__.py", line 166, in __init__
    mod = importlib.import_module(self.SETTINGS_MODULE)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap>", line 1204, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1176, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1147, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 690, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 940, in exec_module
  File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
  File "/home/runner/work/VoyageurCompass/VoyageurCompass/VoyageurCompass/settings.py", line 312, in <module>
    raise ImproperlyConfigured(
django.core.exceptions.ImproperlyConfigured: CORS_ALLOWED_ORIGINS must be set in production!
Error: Process completed with exit code 1.

# frontend-tests
## Run frontend linting:

Run cd Design/frontend

> frontend@0.0.0 lint
> eslint .


/home/runner/work/VoyageurCompass/VoyageurCompass/Design/frontend/src/features/api/apiSlice.test.js
Error:    11:1   error  'global' is not defined                                                                       no-undef
Error:    43:7   error  'global' is not defined                                                                       no-undef
Error:    63:15  error  'result' is assigned a value but never used. Allowed unused vars must match /^[A-Z_]/u        no-unused-vars
Error:    70:16  error  'global' is not defined                                                                       no-undef
Error:    84:7   error  'global' is not defined                                                                       no-undef
Error:    96:15  error  'result' is assigned a value but never used. Allowed unused vars must match /^[A-Z_]/u        no-unused-vars
Error:   103:27  error  'global' is not defined                                                                       no-undef
Error:   129:7   error  'global' is not defined                                                                       no-undef
Error:   146:7   error  'global' is not defined                                                                       no-undef
Error:   169:7   error  'global' is not defined                                                                       no-undef
Error:   204:7   error  'global' is not defined                                                                       no-undef
Error:   212:7   error  'global' is not defined                                                                       no-undef
Error:   219:7   error  'global' is not defined                                                                       no-undef
Error:   244:16  error  'global' is not defined                                                                       no-undef
Error:   255:7   error  'global' is not defined                                                                       no-undef
Error:   263:7   error  'global' is not defined                                                                       no-undef
Error:   270:13  error  'initialState' is assigned a value but never used. Allowed unused vars must match /^[A-Z_]/u  no-unused-vars
Error:   350:7   error  'global' is not defined                                                                       no-undef

/home/runner/work/VoyageurCompass/VoyageurCompass/Design/frontend/vitest.config.js
Error:   76:25  error  '__dirname' is not defined  no-undef
Error:   77:35  error  '__dirname' is not defined  no-undef
Error:   78:30  error  '__dirname' is not defined  no-undef
Error:   79:33  error  '__dirname' is not defined  no-undef
Error:   80:30  error  '__dirname' is not defined  no-undef
Error:   81:30  error  '__dirname' is not defined  no-undef
Error:   82:29  error  '__dirname' is not defined  no-undef

✖ 25 problems (25 errors, 0 warnings)

Error: Process completed with exit code 1.

# security-scan
## Run Bandit security scan:

Run bandit -r . -x tests/,Design/frontend/
[main]	INFO	profile include tests: None
[main]	INFO	profile exclude tests: None
[main]	INFO	cli include tests: None
[main]	INFO	cli exclude tests: None
[main]	INFO	running on Python 3.11.13
Working... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:01
Run started:2025-08-12 00:49:50.408647

Test results:
>> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded password: 'testpass123'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b106_hardcoded_password_funcarg.html
   Location: ./Analytics/tests/test_api.py:24:20
23	        # Create test user
24	        self.user = User.objects.create_user(
25	            username='testuser',
26	            email='test@example.com',
27	            password='testpass123'
28	        )
29	        

--------------------------------------------------
>> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded password: 'testpass123'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b106_hardcoded_password_funcarg.html
   Location: ./Analytics/tests/test_views.py:55:20
54	        """Set up test-specific data."""
55	        self.user = User.objects.create_user(
56	            username='testuser',
57	            password='testpass123'
58	        )
59	        

--------------------------------------------------
>> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded password: 'otherpass123'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b106_hardcoded_password_funcarg.html
   Location: ./Analytics/tests/test_views.py:140:21
139	        """Test that users cannot access other users' portfolio analysis."""
140	        other_user = User.objects.create_user(
141	            username='otheruser',
142	            password='otherpass123'
143	        )
144	        other_portfolio = Portfolio.objects.create(

--------------------------------------------------
>> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded password: 'testpass123'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b106_hardcoded_password_funcarg.html
   Location: ./Analytics/tests/test_views.py:533:20
532	        self.client = APIClient()
533	        self.user = User.objects.create_user(
534	            username='testuser',
535	            password='testpass123'
536	        )
537	        self.client.force_authenticate(user=self.user)

--------------------------------------------------
>> Issue: [B324:hashlib] Use of weak MD5 hash for security. Consider usedforsecurity=False
   Severity: High   Confidence: High
   CWE: CWE-327 (https://cwe.mitre.org/data/definitions/327.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b324_hashlib.html
   Location: ./Core/services/utils.py:148:11
147	    key_string = ':'.join(key_parts)
148	    return hashlib.md5(key_string.encode()).hexdigest()
149	

--------------------------------------------------
>> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded password: 'testpass123'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b106_hardcoded_password_funcarg.html
   Location: ./Core/tests/test_views.py:311:20
310	        self.client = APIClient()
311	        self.user = User.objects.create_user(
312	            username='testuser',
313	            password='testpass123',
314	            email='test@example.com'
315	        )
316	        self.admin_user = User.objects.create_superuser(

--------------------------------------------------
>> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded password: 'adminpass123'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b106_hardcoded_password_funcarg.html
   Location: ./Core/tests/test_views.py:316:26
315	        )
316	        self.admin_user = User.objects.create_superuser(
317	            username='admin',
318	            password='adminpass123',
319	            email='admin@example.com'
320	        )
321	    

--------------------------------------------------
>> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded password: 'testpass123'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b106_hardcoded_password_funcarg.html
   Location: ./Core/tests/test_views.py:457:15
456	        """Test JWT token security features."""
457	        user = User.objects.create_user(
458	            username='testuser',
459	            password='testpass123'
460	        )
461	        

--------------------------------------------------
>> Issue: [B112:try_except_continue] Try, Except, Continue detected.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b112_try_except_continue.html
   Location: ./Data/services/data_processor.py:220:16
219	                    date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
220	                except:
221	                    continue
222	                

--------------------------------------------------
>> Issue: [B311:blacklist] Standard pseudo-random generators are not suitable for security/cryptographic purposes.
   Severity: Low   Confidence: High
   CWE: CWE-330 (https://cwe.mitre.org/data/definitions/330.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/blacklists/blacklist_calls.html#b311-random
   Location: ./Data/services/provider.py:43:59
42	            if retries > 0:
43	                delay = self.base_delay * (3 ** retries) + random.uniform(1, 3)
44	                logger.info(f"Waiting {delay:.1f} seconds before retry...")

--------------------------------------------------
>> Issue: [B311:blacklist] Standard pseudo-random generators are not suitable for security/cryptographic purposes.
   Severity: Low   Confidence: High
   CWE: CWE-330 (https://cwe.mitre.org/data/definitions/330.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/blacklists/blacklist_calls.html#b311-random
   Location: ./Data/services/provider.py:48:27
47	                # Longer initial delay to avoid rate limiting
48	                time.sleep(random.uniform(2, 4))
49	            

--------------------------------------------------
>> Issue: [B311:blacklist] Standard pseudo-random generators are not suitable for security/cryptographic purposes.
   Severity: Low   Confidence: High
   CWE: CWE-330 (https://cwe.mitre.org/data/definitions/330.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/blacklists/blacklist_calls.html#b311-random
   Location: ./Data/services/provider.py:186:24
185	            if i > 0:
186	                delay = random.uniform(1, 3)
187	                logger.info(f"Waiting {delay:.1f} seconds before next request...")

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:38:8
37	        
38	        assert stock.symbol == 'AAPL'
39	        assert stock.short_name == 'Apple Inc.'

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:39:8
38	        assert stock.symbol == 'AAPL'
39	        assert stock.short_name == 'Apple Inc.'
40	        assert stock.sector == 'Technology'

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:40:8
39	        assert stock.short_name == 'Apple Inc.'
40	        assert stock.sector == 'Technology'
41	        assert stock.is_active

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:41:8
40	        assert stock.sector == 'Technology'
41	        assert stock.is_active
42	        assert str(stock) == 'AAPL - Apple Inc.'

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:42:8
41	        assert stock.is_active
42	        assert str(stock) == 'AAPL - Apple Inc.'
43	    

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:77:8
76	        result = stock.get_latest_price()
77	        assert result == latest_price
78	        assert result.close == Decimal('159.00')

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:78:8
77	        assert result == latest_price
78	        assert result.close == Decimal('159.00')
79	    

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:98:8
97	        history = stock.get_price_history(days=7)
98	        assert history.count() == 8
99	        

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:102:8
101	        history = stock.get_price_history(days=30)
102	        assert history.count() == 10
103	    

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:109:8
108	        # No last_sync, should need sync
109	        assert stock.needs_sync is True
110	        

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:114:8
113	        stock.save()
114	        assert stock.needs_sync is False
115	        

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:119:8
118	        stock.save()
119	        assert stock.needs_sync is True
120	

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:140:8
139	        
140	        assert price.stock == stock
141	        assert price.open == Decimal('150.00')

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:141:8
140	        assert price.stock == stock
141	        assert price.open == Decimal('150.00')
142	        assert price.close == Decimal('154.00')

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:142:8
141	        assert price.open == Decimal('150.00')
142	        assert price.close == Decimal('154.00')
143	        assert str(price) == f'AAPL - {date.today()}: $154.00'

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:143:8
142	        assert price.close == Decimal('154.00')
143	        assert str(price) == f'AAPL - {date.today()}: $154.00'
144	    

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:183:8
182	        
183	        assert price.daily_change == Decimal('4.00')
184	    

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:199:8
198	        expected_percent = (Decimal('4.00') / Decimal('150.00')) * Decimal('100')
199	        assert abs(price.daily_change_percent - expected_percent) < Decimal('0.01')
200	    

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:214:8
213	        
214	        assert price.daily_range == '149.00 - 155.00'
215	    

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:230:8
229	        )
230	        assert gain_price.is_gain is True
231	        

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:242:8
241	        )
242	        assert loss_price.is_gain is False
243	

--------------------------------------------------
>> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded password: 'testpass'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b106_hardcoded_password_funcarg.html
   Location: ./Data/tests/test_models.py:251:15
250	        """Test creating a portfolio instance."""
251	        user = User.objects.create_user(username='testuser', password='testpass')
252	        portfolio = Portfolio.objects.create(

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:261:8
260	        
261	        assert portfolio.user == user
262	        assert portfolio.name == 'My Portfolio'

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:262:8
261	        assert portfolio.user == user
262	        assert portfolio.name == 'My Portfolio'
263	        assert portfolio.initial_value == Decimal('10000.00')

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:263:8
262	        assert portfolio.name == 'My Portfolio'
263	        assert portfolio.initial_value == Decimal('10000.00')
264	        assert portfolio.risk_tolerance == 'moderate'

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:264:8
263	        assert portfolio.initial_value == Decimal('10000.00')
264	        assert portfolio.risk_tolerance == 'moderate'
265	        assert str(portfolio) == 'My Portfolio'

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:265:8
264	        assert portfolio.risk_tolerance == 'moderate'
265	        assert str(portfolio) == 'My Portfolio'
266	    

--------------------------------------------------
>> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded password: 'testpass'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b106_hardcoded_password_funcarg.html
   Location: ./Data/tests/test_models.py:269:15
268	        """Test calculating portfolio returns."""
269	        user = User.objects.create_user(username='testuser', password='testpass')
270	        portfolio = Portfolio.objects.create(

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:278:8
277	        returns = portfolio.calculate_returns()
278	        assert returns == Decimal('20.00')  # 20% return
279	    

--------------------------------------------------
>> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded password: 'testpass'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b106_hardcoded_password_funcarg.html
   Location: ./Data/tests/test_models.py:282:15
281	        """Test updating portfolio value based on holdings."""
282	        user = User.objects.create_user(username='testuser', password='testpass')
283	        portfolio = Portfolio.objects.create(

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:315:8
314	        # 10 * 160 + 5 * 320 = 1600 + 1600 = 3200
315	        assert portfolio.current_value == Decimal('3200.00')
316	

--------------------------------------------------
>> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded password: 'testpass'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b106_hardcoded_password_funcarg.html
   Location: ./Data/tests/test_models.py:324:15
323	        """Test creating a portfolio holding."""
324	        user = User.objects.create_user(username='testuser', password='testpass')
325	        portfolio = Portfolio.objects.create(

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:340:8
339	        
340	        assert holding.portfolio == portfolio
341	        assert holding.stock == stock

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:341:8
340	        assert holding.portfolio == portfolio
341	        assert holding.stock == stock
342	        assert holding.quantity == Decimal('10')

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:342:8
341	        assert holding.stock == stock
342	        assert holding.quantity == Decimal('10')
343	        assert str(holding) == 'My Portfolio - AAPL: 10 shares'

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:343:8
342	        assert holding.quantity == Decimal('10')
343	        assert str(holding) == 'My Portfolio - AAPL: 10 shares'
344	    

--------------------------------------------------
>> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded password: 'testpass'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b106_hardcoded_password_funcarg.html
   Location: ./Data/tests/test_models.py:347:15
346	        """Test that derived fields are calculated automatically."""
347	        user = User.objects.create_user(username='testuser', password='testpass')
348	        portfolio = Portfolio.objects.create(

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:364:8
363	        # Check calculated fields
364	        assert holding.cost_basis == Decimal('1500.00')  # 10 * 150
365	        assert holding.current_value == Decimal('1600.00')  # 10 * 160

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:365:8
364	        assert holding.cost_basis == Decimal('1500.00')  # 10 * 150
365	        assert holding.current_value == Decimal('1600.00')  # 10 * 160
366	        assert holding.unrealized_gain_loss == Decimal('100.00')  # 1600 - 1500

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:366:8
365	        assert holding.current_value == Decimal('1600.00')  # 10 * 160
366	        assert holding.unrealized_gain_loss == Decimal('100.00')  # 1600 - 1500
367	        assert abs(holding.unrealized_gain_loss_percent - Decimal('6.67')) < Decimal('0.01')

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:367:8
366	        assert holding.unrealized_gain_loss == Decimal('100.00')  # 1600 - 1500
367	        assert abs(holding.unrealized_gain_loss_percent - Decimal('6.67')) < Decimal('0.01')
368	    

--------------------------------------------------
>> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded password: 'testpass'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b106_hardcoded_password_funcarg.html
   Location: ./Data/tests/test_models.py:371:15
370	        """Test that portfolio-stock combination must be unique."""
371	        user = User.objects.create_user(username='testuser', password='testpass')
372	        portfolio = Portfolio.objects.create(

--------------------------------------------------
>> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded password: 'testpass123'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b106_hardcoded_password_funcarg.html
   Location: ./Data/tests/test_views.py:22:20
21	        """Set up test data."""
22	        self.user = User.objects.create_user(
23	            username='testuser',
24	            password='testpass123'
25	        )
26	        

--------------------------------------------------
>> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded password: 'testpass123'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b106_hardcoded_password_funcarg.html
   Location: ./Data/tests/test_views.py:166:20
165	        """Set up test data."""
166	        self.user = User.objects.create_user(
167	            username='testuser',
168	            password='testpass123'
169	        )
170	        self.client.force_authenticate(user=self.user)

--------------------------------------------------
>> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded password: 'testpass123'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b106_hardcoded_password_funcarg.html
   Location: ./Data/tests/test_views.py:389:20
388	        """Set up test data."""
389	        self.user = User.objects.create_user(
390	            username='testuser',
391	            password='testpass123'
392	        )
393	        

--------------------------------------------------
>> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded password: 'testpass123'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b106_hardcoded_password_funcarg.html
   Location: ./Data/tests/test_views.py:510:20
509	        """Set up test data."""
510	        self.user = User.objects.create_user(
511	            username='testuser',
512	            password='testpass123'
513	        )
514	        self.other_user = User.objects.create_user(

--------------------------------------------------
>> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded password: 'otherpass123'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b106_hardcoded_password_funcarg.html
   Location: ./Data/tests/test_views.py:514:26
513	        )
514	        self.other_user = User.objects.create_user(
515	            username='otheruser',
516	            password='otherpass123'
517	        )
518	        self.client.force_authenticate(user=self.user)

--------------------------------------------------
>> Issue: [B105:hardcoded_password_string] Possible hardcoded password: 'django-insecure-dev-only-key-replace-in-production'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b105_hardcoded_password_string.html
   Location: ./VoyageurCompass/settings.py:53:31
52	SECRET_KEY = env('SECRET_KEY', default='django-insecure-dev-only-key-replace-in-production')
53	if not DEBUG and SECRET_KEY == 'django-insecure-dev-only-key-replace-in-production':
54	    raise ImproperlyConfigured(

--------------------------------------------------
>> Issue: [B404:blacklist] Consider possible security implications associated with the subprocess module.
   Severity: Low   Confidence: High
   CWE: CWE-78 (https://cwe.mitre.org/data/definitions/78.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/blacklists/blacklist_imports.html#b404-import-subprocess
   Location: ./scripts/run_tests.py:9:0
8	import sys
9	import subprocess
10	import argparse

--------------------------------------------------
>> Issue: [B603:subprocess_without_shell_equals_true] subprocess call - check for execution of untrusted input.
   Severity: Low   Confidence: High
   CWE: CWE-78 (https://cwe.mitre.org/data/definitions/78.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b603_subprocess_without_shell_equals_true.html
   Location: ./scripts/run_tests.py:56:21
55	        try:
56	            result = subprocess.run(cmd, check=True)
57	            print("✅ Backend tests passed!")

--------------------------------------------------
>> Issue: [B603:subprocess_without_shell_equals_true] subprocess call - check for execution of untrusted input.
   Severity: Low   Confidence: High
   CWE: CWE-78 (https://cwe.mitre.org/data/definitions/78.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b603_subprocess_without_shell_equals_true.html
   Location: ./scripts/run_tests.py:79:21
78	        try:
79	            result = subprocess.run(cmd, check=True)
80	            print("✅ Frontend tests passed!")

--------------------------------------------------
>> Issue: [B607:start_process_with_partial_path] Starting a process with a partial executable path
   Severity: Low   Confidence: High
   CWE: CWE-78 (https://cwe.mitre.org/data/definitions/78.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b607_start_process_with_partial_path.html
   Location: ./scripts/run_tests.py:94:12
93	            os.chdir(self.project_root)
94	            subprocess.run(["flake8", "."], check=True)
95	            print("✅ Backend linting passed!")

--------------------------------------------------
>> Issue: [B603:subprocess_without_shell_equals_true] subprocess call - check for execution of untrusted input.
   Severity: Low   Confidence: High
   CWE: CWE-78 (https://cwe.mitre.org/data/definitions/78.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b603_subprocess_without_shell_equals_true.html
   Location: ./scripts/run_tests.py:94:12
93	            os.chdir(self.project_root)
94	            subprocess.run(["flake8", "."], check=True)
95	            print("✅ Backend linting passed!")

--------------------------------------------------
>> Issue: [B607:start_process_with_partial_path] Starting a process with a partial executable path
   Severity: Low   Confidence: High
   CWE: CWE-78 (https://cwe.mitre.org/data/definitions/78.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b607_start_process_with_partial_path.html
   Location: ./scripts/run_tests.py:104:12
103	            os.chdir(self.frontend_dir)
104	            subprocess.run(["npm", "run", "lint"], check=True)
105	            print("✅ Frontend linting passed!")

--------------------------------------------------
>> Issue: [B603:subprocess_without_shell_equals_true] subprocess call - check for execution of untrusted input.
   Severity: Low   Confidence: High
   CWE: CWE-78 (https://cwe.mitre.org/data/definitions/78.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b603_subprocess_without_shell_equals_true.html
   Location: ./scripts/run_tests.py:104:12
103	            os.chdir(self.frontend_dir)
104	            subprocess.run(["npm", "run", "lint"], check=True)
105	            print("✅ Frontend linting passed!")

--------------------------------------------------
>> Issue: [B405:blacklist] Using xml.etree.ElementTree to parse untrusted XML data is known to be vulnerable to XML attacks. Replace xml.etree.ElementTree with the equivalent defusedxml package, or make sure defusedxml.defuse_stdlib() is called.
   Severity: Low   Confidence: High
   CWE: CWE-20 (https://cwe.mitre.org/data/definitions/20.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/blacklists/blacklist_imports.html#b405-import-xml-etree
   Location: ./scripts/run_tests.py:131:12
130	        try:
131	            import xml.etree.ElementTree as ET
132	            

--------------------------------------------------
>> Issue: [B314:blacklist] Using xml.etree.ElementTree.parse to parse untrusted XML data is known to be vulnerable to XML attacks. Replace xml.etree.ElementTree.parse with its defusedxml equivalent function or make sure defusedxml.defuse_stdlib() is called
   Severity: Medium   Confidence: High
   CWE: CWE-20 (https://cwe.mitre.org/data/definitions/20.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/blacklists/blacklist_calls.html#b313-b320-xml-bad-elementtree
   Location: ./scripts/run_tests.py:136:23
135	            if backend_xml.exists():
136	                tree = ET.parse(backend_xml)
137	                root = tree.getroot()

--------------------------------------------------
>> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded password: 'otherpass'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b106_hardcoded_password_funcarg.html
   Location: ./tests/test_integration.py:325:21
324	        User.objects.create_user(**self.user_data)
325	        other_user = User.objects.create_user(
326	            username='otheruser',
327	            password='otherpass'
328	        )
329	        

--------------------------------------------------
>> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded password: 'perfpass123'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b106_hardcoded_password_funcarg.html
   Location: ./tests/test_integration.py:474:20
473	        self.client = APIClient()
474	        self.user = User.objects.create_user(
475	            username='perfuser',
476	            password='perfpass123'
477	        )
478	        self.client.force_authenticate(user=self.user)

--------------------------------------------------

Code scanned:
	Total lines of code: 13659
	Total lines skipped (#nosec): 0
	Total potential issues skipped due to specifically being disabled (e.g., #nosec BXXX): 0

Run metrics:
	Total issues (by severity):
		Undefined: 0
		Low: 69
		Medium: 1
		High: 1
	Total issues (by confidence):
		Undefined: 0
		Low: 0
		Medium: 21
		High: 50
Files skipped (4):
	./Analytics/test/test_ta_engine.py (syntax error while parsing AST from file)
	./Analytics/views.py (syntax error while parsing AST from file)
	./Data/repo/price_reader.py (syntax error while parsing AST from file)
	./Data/services/yahoo_finance.py (syntax error while parsing AST from file)
Error: Process completed with exit code 1.

# code-quality
## Run Black formatting check:

# Lots of code, the result:
Oh no! 💥 💔 💥
83 files would be reformatted, 16 files would be left unchanged, 4 files would fail to reformat.
Error: Process completed with exit code 123.

# notify
## Notify failure:

Run echo "❌ Tests failed!"
❌ Tests failed!
Error: Process completed with exit code 1.