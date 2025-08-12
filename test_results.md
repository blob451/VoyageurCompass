# backend-tests
## Run Django check stage:

Run python manage.py migrate
timestamp=2025-08-11T18:17:09 level=INFO logger=Data.services.provider environment="development" message="Data Provider Service initialized"
timestamp=2025-08-11T18:17:09 level=INFO logger=Data.services.synchronizer environment="development" message="Data Synchronizer Service initialized"
timestamp=2025-08-11T18:17:09 level=INFO logger=Data.services.yahoo_finance environment="development" message="Yahoo Finance Service initialized with yfinance integration"
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/django/db/backends/base/base.py", line 279, in ensure_connection
    self.connect()
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/django/utils/asyncio.py", line 26, in inner
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/django/db/backends/base/base.py", line 256, in connect
    self.connection = self.get_new_connection(conn_params)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/django/utils/asyncio.py", line 26, in inner
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/django/db/backends/postgresql/base.py", line 332, in get_new_connection
    connection = self.Database.connect(**conn_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/psycopg2/__init__.py", line 122, in connect
    conn = _connect(dsn, connection_factory=connection_factory, **kwasync)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
psycopg2.OperationalError: connection to server at "localhost" (::1), port 5432 failed: FATAL:  password authentication failed for user "voyageur_user"


The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/runner/work/VoyageurCompass/VoyageurCompass/manage.py", line 22, in <module>
    main()
  File "/home/runner/work/VoyageurCompass/VoyageurCompass/manage.py", line 18, in main
    execute_from_command_line(sys.argv)
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/django/core/management/__init__.py", line 442, in execute_from_command_line
    utility.execute()
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/django/core/management/__init__.py", line 436, in execute
    self.fetch_command(subcommand).run_from_argv(self.argv)
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/django/core/management/base.py", line 416, in run_from_argv
    self.execute(*args, **cmd_options)
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/django/core/management/base.py", line 460, in execute
    output = self.handle(*args, **options)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/django/core/management/base.py", line 107, in wrapper
    res = handle_func(*args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/django/core/management/commands/migrate.py", line 114, in handle
    executor = MigrationExecutor(connection, self.migration_progress_callback)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/django/db/migrations/executor.py", line 18, in __init__
    self.loader = MigrationLoader(self.connection)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/django/db/migrations/loader.py", line 58, in __init__
    self.build_graph()
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/django/db/migrations/loader.py", line 235, in build_graph
    self.applied_migrations = recorder.applied_migrations()
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/django/db/migrations/recorder.py", line 89, in applied_migrations
    if self.has_table():
       ^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/django/db/migrations/recorder.py", line 63, in has_table
    with self.connection.cursor() as cursor:
         ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/django/utils/asyncio.py", line 26, in inner
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/django/db/backends/base/base.py", line 320, in cursor
    return self._cursor()
           ^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/django/db/backends/base/base.py", line 296, in _cursor
    self.ensure_connection()
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/django/utils/asyncio.py", line 26, in inner
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/django/db/backends/base/base.py", line 278, in ensure_connection
    with self.wrap_database_errors:
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/django/db/utils.py", line 91, in __exit__
    raise dj_exc_value.with_traceback(traceback) from exc_value
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/django/db/backends/base/base.py", line 279, in ensure_connection
    self.connect()
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/django/utils/asyncio.py", line 26, in inner
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/django/db/backends/base/base.py", line 256, in connect
    self.connection = self.get_new_connection(conn_params)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/django/utils/asyncio.py", line 26, in inner
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/django/db/backends/postgresql/base.py", line 332, in get_new_connection
    connection = self.Database.connect(**conn_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.13/x64/lib/python3.11/site-packages/psycopg2/__init__.py", line 122, in connect
    conn = _connect(dsn, connection_factory=connection_factory, **kwasync)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
django.db.utils.OperationalError: connection to server at "localhost" (::1), port 5432 failed: FATAL:  password authentication failed for user "voyageur_user"

Error: Process completed with exit code 1.

# frontend-tests
## Run frontend linting:

Run cd Design/frontend

> frontend@0.0.0 test:coverage
> vitest --coverage

 MISSING DEPENDENCY  Cannot find dependency '@vitest/coverage-v8'


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
Run started:2025-08-12 01:16:09.502029

Test results:
>> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded password: 'testpass123'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b106_hardcoded_password_funcarg.html
   Location: ./Analytics/tests/test_api.py:24:20
23	        # Create test user
24	        self.user = User.objects.create_user(
25	            username="testuser", email="test@example.com", ***
26	        )
27	

--------------------------------------------------
>> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded password: 'testpass123'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b106_hardcoded_password_funcarg.html
   Location: ./Analytics/tests/test_views.py:55:20
54	        """Set up test-specific data."""
55	        self.user = User.objects.create_user(
56	            username="testuser", ***
57	        )
58	

--------------------------------------------------
>> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded password: 'otherpass123'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b106_hardcoded_password_funcarg.html
   Location: ./Analytics/tests/test_views.py:139:21
138	        """Test that users cannot access other users' portfolio analysis."""
139	        other_user = User.objects.create_user(
140	            username="otheruser", ***
141	        )
142	        other_portfolio = Portfolio.objects.create(

--------------------------------------------------
>> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded password: 'testpass123'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b106_hardcoded_password_funcarg.html
   Location: ./Analytics/tests/test_views.py:505:20
504	        self.client = APIClient()
505	        self.user = User.objects.create_user(
506	            username="testuser", ***
507	        )
508	        self.client.force_authenticate(user=self.user)

--------------------------------------------------
>> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded password: 'testpass123'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b106_hardcoded_password_funcarg.html
   Location: ./Core/tests/test_views.py:307:20
306	        self.client = APIClient()
307	        self.user = User.objects.create_user(
308	            username="testuser", ***, email="test@example.com"
309	        )
310	        self.admin_user = User.objects.create_superuser(

--------------------------------------------------
>> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded password: 'adminpass123'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b106_hardcoded_password_funcarg.html
   Location: ./Core/tests/test_views.py:310:26
309	        )
310	        self.admin_user = User.objects.create_superuser(
311	            username="admin", ***, email="admin@example.com"
312	        )
313	

--------------------------------------------------
>> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded password: 'testpass123'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b106_hardcoded_password_funcarg.html
   Location: ./Core/tests/test_views.py:465:15
464	        """Test JWT token security features."""
465	        user = User.objects.create_user(username="testuser", ***)
466	

--------------------------------------------------
>> Issue: [B112:try_except_continue] Try, Except, Continue detected.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b112_try_except_continue.html
   Location: ./Data/services/data_processor.py:223:16
222	                    date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
223	                except:
224	                    continue
225	

--------------------------------------------------
>> Issue: [B311:blacklist] Standard pseudo-random generators are not suitable for security/cryptographic purposes.
   Severity: Low   Confidence: High
   CWE: CWE-330 (https://cwe.mitre.org/data/definitions/330.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/blacklists/blacklist_calls.html#b311-random
   Location: ./Data/services/provider.py:43:57
42	            if retries > 0:
43	                delay = self.base_delay * (3**retries) + random.uniform(1, 3)
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
   Location: ./Data/services/provider.py:197:24
196	            if i > 0:
197	                delay = random.uniform(1, 3)
198	                logger.info(f"Waiting {delay:.1f} seconds before next request...")

--------------------------------------------------
>> Issue: [B311:blacklist] Standard pseudo-random generators are not suitable for security/cryptographic purposes.
   Severity: Low   Confidence: High
   CWE: CWE-330 (https://cwe.mitre.org/data/definitions/330.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/blacklists/blacklist_calls.html#b311-random
   Location: ./Data/services/yahoo_finance.py:703:60
702	                        delay = min(
703	                            self.baseDelay * (2**attempt) + random.uniform(0, 1),
704	                            self.maxBackoff,

--------------------------------------------------
>> Issue: [B311:blacklist] Standard pseudo-random generators are not suitable for security/cryptographic purposes.
   Severity: Low   Confidence: High
   CWE: CWE-330 (https://cwe.mitre.org/data/definitions/330.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/blacklists/blacklist_calls.html#b311-random
   Location: ./Data/services/yahoo_finance.py:732:27
731	            if i > 0:
732	                time.sleep(random.uniform(1, 2))
733	

--------------------------------------------------
>> Issue: [B311:blacklist] Standard pseudo-random generators are not suitable for security/cryptographic purposes.
   Severity: Low   Confidence: High
   CWE: CWE-330 (https://cwe.mitre.org/data/definitions/330.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/blacklists/blacklist_calls.html#b311-random
   Location: ./Data/services/yahoo_finance.py:1032:31
1031	                if i > 0:
1032	                    time.sleep(random.uniform(1, 2))
1033	

--------------------------------------------------
>> Issue: [B110:try_except_pass] Try, Except, Pass detected.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b110_try_except_pass.html
   Location: ./Data/services/yahoo_finance.py:1735:24
1734	                            created_count += 1
1735	                        except Exception:
1736	                            pass  # Skip conflicts
1737	

--------------------------------------------------
>> Issue: [B110:try_except_pass] Try, Except, Pass detected.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b110_try_except_pass.html
   Location: ./Data/services/yahoo_finance.py:1757:24
1756	                            created_count += 1
1757	                        except Exception:
1758	                            pass  # Skip errors
1759	

--------------------------------------------------
>> Issue: [B110:try_except_pass] Try, Except, Pass detected.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b110_try_except_pass.html
   Location: ./Data/services/yahoo_finance.py:1824:12
1823	                connection.close()
1824	            except Exception:
1825	                pass  # Ignore connection close errors
1826	

--------------------------------------------------
>> Issue: [B311:blacklist] Standard pseudo-random generators are not suitable for security/cryptographic purposes.
   Severity: Low   Confidence: High
   CWE: CWE-330 (https://cwe.mitre.org/data/definitions/330.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/blacklists/blacklist_calls.html#b311-random
   Location: ./Data/services/yahoo_finance.py:2490:47
2489	                    if attempt < max_attempts:
2490	                        delay = (2**attempt) + random.uniform(0, 1)
2491	                        logger.info(f"Waiting {delay:.2f}s before retry...")

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:38:8
37	
38	        assert stock.symbol == "AAPL"
39	        assert stock.short_name == "Apple Inc."

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:39:8
38	        assert stock.symbol == "AAPL"
39	        assert stock.short_name == "Apple Inc."
40	        assert stock.sector == "Technology"

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:40:8
39	        assert stock.short_name == "Apple Inc."
40	        assert stock.sector == "Technology"
41	        assert stock.is_active

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:41:8
40	        assert stock.sector == "Technology"
41	        assert stock.is_active
42	        assert str(stock) == "AAPL - Apple Inc."

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:42:8
41	        assert stock.is_active
42	        assert str(stock) == "AAPL - Apple Inc."
43	

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:77:8
76	        result = stock.get_latest_price()
77	        assert result == latest_price
78	        assert result.close == Decimal("159.00")

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:78:8
77	        assert result == latest_price
78	        assert result.close == Decimal("159.00")
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
141	        assert price.open == Decimal("150.00")

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:141:8
140	        assert price.stock == stock
141	        assert price.open == Decimal("150.00")
142	        assert price.close == Decimal("154.00")

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:142:8
141	        assert price.open == Decimal("150.00")
142	        assert price.close == Decimal("154.00")
143	        assert str(price) == f"AAPL - {date.today()}: $154.00"

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:143:8
142	        assert price.close == Decimal("154.00")
143	        assert str(price) == f"AAPL - {date.today()}: $154.00"
144	

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:183:8
182	
183	        assert price.daily_change == Decimal("4.00")
184	

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:199:8
198	        expected_percent = (Decimal("4.00") / Decimal("150.00")) * Decimal("100")
199	        assert abs(price.daily_change_percent - expected_percent) < Decimal("0.01")
200	

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:214:8
213	
214	        assert price.daily_range == "149.00 - 155.00"
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
251	        user = User.objects.create_user(username="testuser", ***)
252	        portfolio = Portfolio.objects.create(

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:261:8
260	
261	        assert portfolio.user == user
262	        assert portfolio.name == "My Portfolio"

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:262:8
261	        assert portfolio.user == user
262	        assert portfolio.name == "My Portfolio"
263	        assert portfolio.initial_value == Decimal("10000.00")

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:263:8
262	        assert portfolio.name == "My Portfolio"
263	        assert portfolio.initial_value == Decimal("10000.00")
264	        assert portfolio.risk_tolerance == "moderate"

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:264:8
263	        assert portfolio.initial_value == Decimal("10000.00")
264	        assert portfolio.risk_tolerance == "moderate"
265	        assert str(portfolio) == "My Portfolio"

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:265:8
264	        assert portfolio.risk_tolerance == "moderate"
265	        assert str(portfolio) == "My Portfolio"
266	

--------------------------------------------------
>> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded password: 'testpass'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b106_hardcoded_password_funcarg.html
   Location: ./Data/tests/test_models.py:269:15
268	        """Test calculating portfolio returns."""
269	        user = User.objects.create_user(username="testuser", ***)
270	        portfolio = Portfolio.objects.create(

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:278:8
277	        returns = portfolio.calculate_returns()
278	        assert returns == Decimal("20.00")  # 20% return
279	

--------------------------------------------------
>> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded password: 'testpass'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b106_hardcoded_password_funcarg.html
   Location: ./Data/tests/test_models.py:282:15
281	        """Test updating portfolio value based on holdings."""
282	        user = User.objects.create_user(username="testuser", ***)
283	        portfolio = Portfolio.objects.create(

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:313:8
312	        # 10 * 160 + 5 * 320 = 1600 + 1600 = 3200
313	        assert portfolio.current_value == Decimal("3200.00")
314	

--------------------------------------------------
>> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded password: 'testpass'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b106_hardcoded_password_funcarg.html
   Location: ./Data/tests/test_models.py:322:15
321	        """Test creating a portfolio holding."""
322	        user = User.objects.create_user(username="testuser", ***)
323	        portfolio = Portfolio.objects.create(user=user, name="My Portfolio")

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:335:8
334	
335	        assert holding.portfolio == portfolio
336	        assert holding.stock == stock

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:336:8
335	        assert holding.portfolio == portfolio
336	        assert holding.stock == stock
337	        assert holding.quantity == Decimal("10")

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:337:8
336	        assert holding.stock == stock
337	        assert holding.quantity == Decimal("10")
338	        assert str(holding) == "My Portfolio - AAPL: 10 shares"

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:338:8
337	        assert holding.quantity == Decimal("10")
338	        assert str(holding) == "My Portfolio - AAPL: 10 shares"
339	

--------------------------------------------------
>> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded password: 'testpass'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b106_hardcoded_password_funcarg.html
   Location: ./Data/tests/test_models.py:342:15
341	        """Test that derived fields are calculated automatically."""
342	        user = User.objects.create_user(username="testuser", ***)
343	        portfolio = Portfolio.objects.create(user=user, name="My Portfolio")

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:356:8
355	        # Check calculated fields
356	        assert holding.cost_basis == Decimal("1500.00")  # 10 * 150
357	        assert holding.current_value == Decimal("1600.00")  # 10 * 160

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:357:8
356	        assert holding.cost_basis == Decimal("1500.00")  # 10 * 150
357	        assert holding.current_value == Decimal("1600.00")  # 10 * 160
358	        assert holding.unrealized_gain_loss == Decimal("100.00")  # 1600 - 1500

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:358:8
357	        assert holding.current_value == Decimal("1600.00")  # 10 * 160
358	        assert holding.unrealized_gain_loss == Decimal("100.00")  # 1600 - 1500
359	        assert abs(holding.unrealized_gain_loss_percent - Decimal("6.67")) < Decimal(

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b101_assert_used.html
   Location: ./Data/tests/test_models.py:359:8
358	        assert holding.unrealized_gain_loss == Decimal("100.00")  # 1600 - 1500
359	        assert abs(holding.unrealized_gain_loss_percent - Decimal("6.67")) < Decimal(
360	            "0.01"
361	        )
362	

--------------------------------------------------
>> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded password: 'testpass'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b106_hardcoded_password_funcarg.html
   Location: ./Data/tests/test_models.py:365:15
364	        """Test that portfolio-stock combination must be unique."""
365	        user = User.objects.create_user(username="testuser", ***)
366	        portfolio = Portfolio.objects.create(user=user, name="My Portfolio")

--------------------------------------------------
>> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded password: 'testpass123'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b106_hardcoded_password_funcarg.html
   Location: ./Data/tests/test_views.py:22:20
21	        """Set up test data."""
22	        self.user = User.objects.create_user(
23	            username="testuser", ***
24	        )
25	

--------------------------------------------------
>> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded password: 'testpass123'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b106_hardcoded_password_funcarg.html
   Location: ./Data/tests/test_views.py:163:20
162	        """Set up test data."""
163	        self.user = User.objects.create_user(
164	            username="testuser", ***
165	        )
166	        self.client.force_authenticate(user=self.user)

--------------------------------------------------
>> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded password: 'testpass123'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b106_hardcoded_password_funcarg.html
   Location: ./Data/tests/test_views.py:377:20
376	        """Set up test data."""
377	        self.user = User.objects.create_user(
378	            username="testuser", ***
379	        )
380	

--------------------------------------------------
>> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded password: 'testpass123'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b106_hardcoded_password_funcarg.html
   Location: ./Data/tests/test_views.py:488:20
487	        """Set up test data."""
488	        self.user = User.objects.create_user(
489	            username="testuser", ***
490	        )
491	        self.other_user = User.objects.create_user(

--------------------------------------------------
>> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded password: 'otherpass123'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b106_hardcoded_password_funcarg.html
   Location: ./Data/tests/test_views.py:491:26
490	        )
491	        self.other_user = User.objects.create_user(
492	            username="otheruser", ***
493	        )
494	        self.client.force_authenticate(user=self.user)

--------------------------------------------------
>> Issue: [B105:hardcoded_password_string] Possible hardcoded password: 'django-insecure-dev-only-key-replace-in-production'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b105_hardcoded_password_string.html
   Location: ./VoyageurCompass/settings.py:56:31
55	)
56	if not DEBUG and SECRET_KEY == "django-insecure-dev-only-key-replace-in-production":
57	    raise ImproperlyConfigured(

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
   Location: ./scripts/run_tests.py:58:21
57	        try:
58	            result = subprocess.run(cmd, check=True)
59	            print("✅ Backend tests passed!")

--------------------------------------------------
>> Issue: [B603:subprocess_without_shell_equals_true] subprocess call - check for execution of untrusted input.
   Severity: Low   Confidence: High
   CWE: CWE-78 (https://cwe.mitre.org/data/definitions/78.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b603_subprocess_without_shell_equals_true.html
   Location: ./scripts/run_tests.py:81:21
80	        try:
81	            result = subprocess.run(cmd, check=True)
82	            print("✅ Frontend tests passed!")

--------------------------------------------------
>> Issue: [B607:start_process_with_partial_path] Starting a process with a partial executable path
   Severity: Low   Confidence: High
   CWE: CWE-78 (https://cwe.mitre.org/data/definitions/78.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b607_start_process_with_partial_path.html
   Location: ./scripts/run_tests.py:96:12
95	            os.chdir(self.project_root)
96	            subprocess.run(["flake8", "."], check=True)
97	            print("✅ Backend linting passed!")

--------------------------------------------------
>> Issue: [B603:subprocess_without_shell_equals_true] subprocess call - check for execution of untrusted input.
   Severity: Low   Confidence: High
   CWE: CWE-78 (https://cwe.mitre.org/data/definitions/78.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b603_subprocess_without_shell_equals_true.html
   Location: ./scripts/run_tests.py:96:12
95	            os.chdir(self.project_root)
96	            subprocess.run(["flake8", "."], check=True)
97	            print("✅ Backend linting passed!")

--------------------------------------------------
>> Issue: [B607:start_process_with_partial_path] Starting a process with a partial executable path
   Severity: Low   Confidence: High
   CWE: CWE-78 (https://cwe.mitre.org/data/definitions/78.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b607_start_process_with_partial_path.html
   Location: ./scripts/run_tests.py:106:12
105	            os.chdir(self.frontend_dir)
106	            subprocess.run(["npm", "run", "lint"], check=True)
107	            print("✅ Frontend linting passed!")

--------------------------------------------------
>> Issue: [B603:subprocess_without_shell_equals_true] subprocess call - check for execution of untrusted input.
   Severity: Low   Confidence: High
   CWE: CWE-78 (https://cwe.mitre.org/data/definitions/78.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b603_subprocess_without_shell_equals_true.html
   Location: ./scripts/run_tests.py:106:12
105	            os.chdir(self.frontend_dir)
106	            subprocess.run(["npm", "run", "lint"], check=True)
107	            print("✅ Frontend linting passed!")

--------------------------------------------------
>> Issue: [B405:blacklist] Using xml.etree.ElementTree to parse untrusted XML data is known to be vulnerable to XML attacks. Replace xml.etree.ElementTree with the equivalent defusedxml package, or make sure defusedxml.defuse_stdlib() is called.
   Severity: Low   Confidence: High
   CWE: CWE-20 (https://cwe.mitre.org/data/definitions/20.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/blacklists/blacklist_imports.html#b405-import-xml-etree
   Location: ./scripts/run_tests.py:133:12
132	        try:
133	            import xml.etree.ElementTree as ET
134	

--------------------------------------------------
>> Issue: [B314:blacklist] Using xml.etree.ElementTree.parse to parse untrusted XML data is known to be vulnerable to XML attacks. Replace xml.etree.ElementTree.parse with its defusedxml equivalent function or make sure defusedxml.defuse_stdlib() is called
   Severity: Medium   Confidence: High
   CWE: CWE-20 (https://cwe.mitre.org/data/definitions/20.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/blacklists/blacklist_calls.html#b313-b320-xml-bad-elementtree
   Location: ./scripts/run_tests.py:138:23
137	            if backend_xml.exists():
138	                tree = ET.parse(backend_xml)
139	                root = tree.getroot()

--------------------------------------------------
>> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded password: 'otherpass'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b106_hardcoded_password_funcarg.html
   Location: ./tests/test_integration.py:326:21
325	        User.objects.create_user(**self.user_data)
326	        other_user = User.objects.create_user(
327	            username="otheruser", ***
328	        )
329	

--------------------------------------------------
>> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded password: 'perfpass123'
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b106_hardcoded_password_funcarg.html
   Location: ./tests/test_integration.py:471:20
470	        self.client = APIClient()
471	        self.user = User.objects.create_user(
472	            username="perfuser", ***
473	        )
474	        self.client.force_authenticate(user=self.user)

--------------------------------------------------

Code scanned:
	Total lines of code: 15575
	Total lines skipped (#nosec): 0
	Total potential issues skipped due to specifically being disabled (e.g., #nosec BXXX): 0

Run metrics:
	Total issues (by severity):
		Undefined: 0
		Low: 76
		Medium: 1
		High: 0
	Total issues (by confidence):
		Undefined: 0
		Low: 0
		Medium: 21
		High: 56
Files skipped (0):
Error: Process completed with exit code 1.

# code-quality
## Run Black formatting check:

Run isort --check-only --diff .
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/tests/test_integration.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/scripts/run_tests.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Analytics/views.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Analytics/urls.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Analytics/engine/ta_engine.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Analytics/tests/test_views.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Analytics/tests/test_api.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Analytics/test/test_ta_engine.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Analytics/test/test_logging.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Core/views.py Imports are incorrectly sorted and/or formatted.
--- /home/runner/work/VoyageurCompass/VoyageurCompass/tests/test_integration.py:before	2025-08-12 01:15:57.051280
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/tests/test_integration.py:after	2025-08-12 01:16:01.870662
@@ -8,15 +8,16 @@
 # Mark all tests in this module as integration tests
 pytestmark = pytest.mark.integration
 
+from datetime import date, timedelta
+from decimal import Decimal
+from unittest.mock import patch
+
 from django.contrib.auth.models import User
 from django.urls import reverse
 from rest_framework import status
-from rest_framework.test import APITestCase, APIClient
-from datetime import date, timedelta
-from decimal import Decimal
-from unittest.mock import patch
-
-from Data.models import Stock, StockPrice, Portfolio, PortfolioHolding
+from rest_framework.test import APIClient, APITestCase
+
+from Data.models import Portfolio, PortfolioHolding, Stock, StockPrice
 
 
 class FullWorkflowIntegrationTest(APITestCase):
--- /home/runner/work/VoyageurCompass/VoyageurCompass/scripts/run_tests.py:before	2025-08-12 01:15:57.051280
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/scripts/run_tests.py:after	2025-08-12 01:16:01.875246
@@ -4,10 +4,10 @@
 Runs both backend and frontend tests with coverage reporting.
 """
 
+import argparse
 import os
+import subprocess
 import sys
-import subprocess
-import argparse
 from pathlib import Path
 
 
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Analytics/views.py:before	2025-08-12 01:15:57.043279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Analytics/views.py:after	2025-08-12 01:16:01.884302
@@ -7,17 +7,19 @@
 """
 
 from datetime import datetime
+
+from drf_spectacular.types import OpenApiTypes
+from drf_spectacular.utils import OpenApiParameter, extend_schema
 from rest_framework import status
-from rest_framework.decorators import api_view, permission_classes, throttle_classes
+from rest_framework.decorators import (api_view, permission_classes,
+                                       throttle_classes)
+from rest_framework.permissions import AllowAny, IsAuthenticated
 from rest_framework.response import Response
-from rest_framework.permissions import IsAuthenticated, AllowAny
-from rest_framework.throttling import UserRateThrottle, AnonRateThrottle
-from drf_spectacular.utils import extend_schema, OpenApiParameter
-from drf_spectacular.types import OpenApiTypes
+from rest_framework.throttling import AnonRateThrottle, UserRateThrottle
 
 from Analytics.engine.ta_engine import TechnicalAnalysisEngine
+from Data.models import Portfolio
 from Data.services.yahoo_finance import yahoo_finance_service
-from Data.models import Portfolio
 
 
 class AnalysisThrottle(UserRateThrottle):
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Analytics/urls.py:before	2025-08-12 01:15:57.043279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Analytics/urls.py:after	2025-08-12 01:16:01.886025
@@ -3,12 +3,9 @@
 """
 
 from django.urls import path
-from Analytics.views import (
-    analyze_stock,
-    analyze_portfolio,
-    batch_analysis,
-    market_overview,
-)
+
+from Analytics.views import (analyze_portfolio, analyze_stock, batch_analysis,
+                             market_overview)
 
 app_name = "analytics"
 
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Analytics/engine/ta_engine.py:before	2025-08-12 01:15:57.042279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Analytics/engine/ta_engine.py:after	2025-08-12 01:16:01.897336
@@ -7,21 +7,17 @@
 """
 
 import logging
-from datetime import datetime, date, timedelta
-from decimal import Decimal
-from typing import Dict, List, Optional, Any, Tuple, NamedTuple
 import math
 import statistics
+from datetime import date, datetime, timedelta
+from decimal import Decimal
+from typing import Any, Dict, List, NamedTuple, Optional, Tuple
 
 from django.utils import timezone
 
-from Data.repo.price_reader import (
-    PriceReader,
-    PriceData,
-    SectorPriceData,
-    IndustryPriceData,
-)
 from Data.repo.analytics_writer import AnalyticsWriter
+from Data.repo.price_reader import (IndustryPriceData, PriceData, PriceReader,
+                                    SectorPriceData)
 
 logger = logging.getLogger(__name__)
 
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Analytics/tests/test_views.py:before	2025-08-12 01:15:57.043279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Analytics/tests/test_views.py:after	2025-08-12 01:16:01.907107
@@ -1,19 +1,20 @@
 """
 Comprehensive tests for Analytics app API views.
 """
+
+from datetime import date, datetime, timedelta
+from decimal import Decimal
+from unittest.mock import MagicMock, patch
 
 import pytest
 from django.contrib.auth.models import User
 from django.urls import reverse
 from django.utils import timezone
 from rest_framework import status
-from rest_framework.test import APITestCase, APIClient
-from unittest.mock import patch, MagicMock
-from datetime import datetime, timedelta, date
-from decimal import Decimal
-
-from Data.models import Stock, StockPrice, Portfolio, PortfolioHolding
+from rest_framework.test import APIClient, APITestCase
+
 from Analytics.services.engine import analytics_engine
+from Data.models import Portfolio, PortfolioHolding, Stock, StockPrice
 
 
 class AnalyticsAPITestCase(APITestCase):
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Analytics/tests/test_api.py:before	2025-08-12 01:15:57.042279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Analytics/tests/test_api.py:after	2025-08-12 01:16:01.910853
@@ -3,13 +3,15 @@
 Tests the full API flow including authentication and data validation.
 """
 
+from datetime import date, timedelta
+from decimal import Decimal
+
 import pytest
-from rest_framework.test import APITestCase
-from rest_framework import status
 from django.contrib.auth import get_user_model
 from django.urls import reverse
-from datetime import date, timedelta
-from decimal import Decimal
+from rest_framework import status
+from rest_framework.test import APITestCase
+
 from Data.models import Stock, StockPrice
 
 User = get_user_model()
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Analytics/test/test_ta_engine.py:before	2025-08-12 01:15:57.042279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Analytics/test/test_ta_engine.py:after	2025-08-12 01:16:01.918137
@@ -4,21 +4,16 @@
 """
 
 import unittest
-from datetime import datetime, date, timedelta
+from datetime import date, datetime, timedelta
 from decimal import Decimal
+
 from django.test import TestCase
 from django.utils import timezone
 
-from Data.models import (
-    Stock,
-    StockPrice,
-    DataSector,
-    DataIndustry,
-    DataSectorPrice,
-    DataIndustryPrice,
-)
+from Analytics.engine.ta_engine import IndicatorResult, TechnicalAnalysisEngine
+from Data.models import (DataIndustry, DataIndustryPrice, DataSector,
+                         DataSectorPrice, Stock, StockPrice)
 from Data.repo.price_reader import PriceData
-from Analytics.engine.ta_engine import TechnicalAnalysisEngine, IndicatorResult
 
 
 class TechnicalAnalysisEngineTestCase(TestCase):
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Analytics/test/test_logging.py:before	2025-08-12 01:15:57.042279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Analytics/test/test_logging.py:after	2025-08-12 01:16:01.919599
@@ -1,6 +1,7 @@
 import logging
+from io import StringIO
+
 from django.test import TestCase
-from io import StringIO
 
 
 class AnalyticsLoggingTestCase(TestCase):
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Core/views.py:before	2025-08-12 01:15:57.044279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Core/views.py:after	2025-08-12 01:16:01.931883
@@ -4,23 +4,20 @@
 """
 
 import logging
-from rest_framework import status, generics
+
+from django.contrib.auth.models import User
+from django.db import connection
+from drf_spectacular.utils import extend_schema
+from rest_framework import generics, status
 from rest_framework.decorators import api_view, permission_classes
+from rest_framework.permissions import AllowAny, IsAuthenticated
 from rest_framework.response import Response
-from rest_framework.permissions import IsAuthenticated, AllowAny
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Core/urls.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Core/serializers.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Core/middleware.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Core/tests/test_views.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Core/services/utils.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Core/services/auth.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Core/management/commands/core_run_analytics_one.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Core/test/test_sentry.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Core/test/test_health.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Core/test/test_logging.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/VoyageurCompass/urls.py Imports are incorrectly sorted and/or formatted.
 from rest_framework.views import APIView
 from rest_framework_simplejwt.tokens import RefreshToken
 from rest_framework_simplejwt.views import TokenObtainPairView
-from django.contrib.auth.models import User
-from django.db import connection
-from drf_spectacular.utils import extend_schema
-
-from Core.serializers import (
-    UserSerializer,
-    UserRegistrationSerializer,
-    ChangePasswordSerializer,
-    UserProfileSerializer,
-)
+
+from Core.serializers import (ChangePasswordSerializer, UserProfileSerializer,
+                              UserRegistrationSerializer, UserSerializer)
 
 
 class CustomTokenObtainPairView(TokenObtainPairView):
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Core/urls.py:before	2025-08-12 01:15:57.044279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Core/urls.py:after	2025-08-12 01:16:01.934241
@@ -4,16 +4,12 @@
 
 from django.urls import path
 from rest_framework_simplejwt.views import TokenRefreshView
-from Core.views import (
-    CustomTokenObtainPairView,
-    RegisterView,
-    UserProfileView,
-    ChangePasswordView,
-    health_check,
-    user_stats,
-    healthCheck as health_check_liveness,
-    readinessCheck as readiness_check,
-)
+
+from Core.views import (ChangePasswordView, CustomTokenObtainPairView,
+                        RegisterView, UserProfileView, health_check)
+from Core.views import healthCheck as health_check_liveness
+from Core.views import readinessCheck as readiness_check
+from Core.views import user_stats
 
 app_name = "core"
 
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Core/serializers.py:before	2025-08-12 01:15:57.043279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Core/serializers.py:after	2025-08-12 01:16:01.938360
@@ -3,9 +3,9 @@
 Handles user authentication and registration.
 """
 
-from rest_framework import serializers
 from django.contrib.auth.models import User
 from django.contrib.auth.password_validation import validate_password
+from rest_framework import serializers
 from rest_framework.validators import UniqueValidator
 
 
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Core/middleware.py:before	2025-08-12 01:15:57.043279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Core/middleware.py:after	2025-08-12 01:16:01.941138
@@ -1,6 +1,7 @@
+import logging
 import time
 import uuid
-import logging
+
 from django.http import HttpResponse
 from django.utils.cache import patch_vary_headers
 from django.utils.deprecation import MiddlewareMixin
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Core/tests/test_views.py:before	2025-08-12 01:15:57.044279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Core/tests/test_views.py:after	2025-08-12 01:16:01.950216
@@ -1,14 +1,15 @@
 """
 Comprehensive tests for Core app API views.
 """
+
+from datetime import datetime, timedelta
+from unittest.mock import MagicMock, patch
 
 import pytest
 from django.contrib.auth.models import User
 from django.urls import reverse
 from rest_framework import status
-from rest_framework.test import APITestCase, APIClient
-from unittest.mock import patch, MagicMock
-from datetime import datetime, timedelta
+from rest_framework.test import APIClient, APITestCase
 from rest_framework_simplejwt.tokens import RefreshToken
 
 
@@ -404,6 +405,7 @@
     def test_rate_limiting(self):
         """Test rate limiting on public endpoints."""
         from unittest.mock import patch
+
         from rest_framework.throttling import AnonRateThrottle
 
         url = reverse("core:health-check")
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Core/services/utils.py:before	2025-08-12 01:15:57.043279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Core/services/utils.py:after	2025-08-12 01:16:01.953250
@@ -3,12 +3,12 @@
 Common utility functions used across VoyageurCompass.
 """
 
-import logging
 import hashlib
 import json
+import logging
 from datetime import datetime, timedelta
+from decimal import Decimal, InvalidOperation
 from typing import Any, Dict, List, Optional, Union
-from decimal import Decimal, InvalidOperation
 
 logger = logging.getLogger(__name__)
 
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Core/services/auth.py:before	2025-08-12 01:15:57.043279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Core/services/auth.py:after	2025-08-12 01:16:01.957005
@@ -5,12 +5,13 @@
 
 import logging
 import re
-from typing import Optional, Dict
+from typing import Dict, Optional
+
 from django.contrib.auth import authenticate, login, logout
 from django.contrib.auth.models import User
+from django.contrib.auth.password_validation import validate_password
 from django.core.exceptions import ValidationError
 from django.db import IntegrityError
-from django.contrib.auth.password_validation import validate_password
 
 logger = logging.getLogger(__name__)
 
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Core/management/commands/core_run_analytics_one.py:before	2025-08-12 01:15:57.043279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Core/management/commands/core_run_analytics_one.py:after	2025-08-12 01:16:01.967006
@@ -6,20 +6,20 @@
     python manage.py core_run_analytics_one --symbol AAPL
 """
 
+import logging
 import os
 import uuid
-import logging
 from datetime import datetime, timedelta
-from typing import Dict, Any
+from typing import Any, Dict
 
 from django.core.management.base import BaseCommand, CommandError
+from django.db import connection
 from django.utils import timezone
-from django.db import connection
-
+
+from Analytics.engine.ta_engine import TechnicalAnalysisEngine
+from Data.models import AnalyticsResults, Stock
 from Data.repo.price_reader import PriceReader
 from Data.services.yahoo_finance import yahoo_finance_service
-from Analytics.engine.ta_engine import TechnicalAnalysisEngine
-from Data.models import Stock, AnalyticsResults
 
 logger = logging.getLogger(__name__)
 
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Core/test/test_sentry.py:before	2025-08-12 01:15:57.043279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Core/test/test_sentry.py:after	2025-08-12 01:16:01.969213
@@ -1,7 +1,8 @@
 import os
-from unittest.mock import patch, MagicMock
+from unittest.mock import MagicMock, patch
+
+from django.conf import settings
 from django.test import TestCase, override_settings
-from django.conf import settings
 
 
 class SentryIntegrationTestCase(TestCase):
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Core/test/test_health.py:before	2025-08-12 01:15:57.043279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Core/test/test_health.py:after	2025-08-12 01:16:01.970482
@@ -1,6 +1,7 @@
 from unittest.mock import patch
-from django.test import TestCase, Client
+
 from django.db import connection
+from django.test import Client, TestCase
 
 
 class HealthCheckTestCase(TestCase):
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Core/test/test_logging.py:before	2025-08-12 01:15:57.043279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Core/test/test_logging.py:after	2025-08-12 01:16:01.971883
@@ -1,8 +1,9 @@
 import json
 import logging
 import os
+from io import StringIO
+
 from django.test import TestCase, override_settings
-from io import StringIO
 
 
 class LoggingConfigTestCase(TestCase):
--- /home/runner/work/VoyageurCompass/VoyageurCompass/VoyageurCompass/urls.py:before	2025-08-12 01:15:57.050279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/VoyageurCompass/urls.py:after	2025-08-12 01:16:01.975220
@@ -15,19 +15,15 @@
     2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
 """
 
-from django.contrib import admin
-from django.urls import path, include
 from django.conf import settings
 from django.conf.urls.static import static
-from drf_spectacular.views import (
-    SpectacularAPIView,
-    SpectacularRedocView,
-    SpectacularSwaggerView,
-)
-from Core.views import (
-    healthCheck as health_check_liveness,
-    readinessCheck as readiness_check,
-)
+from django.contrib import admin
+from django.urls import include, path
+from drf_spectacular.views import (SpectacularAPIView, SpectacularRedocView,
+                                   SpectacularSwaggerView)
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/VoyageurCompass/settings.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/VoyageurCompass/celery.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Data/market_views.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Data/managers.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Data/views.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Data/urls.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Data/models.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Data/serializers.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Data/repo/price_reader.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Data/repo/analytics_writer.py Imports are incorrectly sorted and/or formatted.
+
+from Core.views import healthCheck as health_check_liveness
+from Core.views import readinessCheck as readiness_check
 
 urlpatterns = [
     path("admin/", admin.site.urls),
--- /home/runner/work/VoyageurCompass/VoyageurCompass/VoyageurCompass/settings.py:before	2025-08-12 01:15:57.050279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/VoyageurCompass/settings.py:after	2025-08-12 01:16:01.981829
@@ -10,12 +10,13 @@
 https://docs.djangoproject.com/en/5.2/ref/settings/
 """
 
-import environ
+import json
 import os
 import sys
-import json
+from datetime import timedelta
 from pathlib import Path
-from datetime import timedelta
+
+import environ
 from django.core.exceptions import ImproperlyConfigured
 
 # Guard Sentry import to prevent crashes when package not installed
--- /home/runner/work/VoyageurCompass/VoyageurCompass/VoyageurCompass/celery.py:before	2025-08-12 01:15:57.050279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/VoyageurCompass/celery.py:after	2025-08-12 01:16:01.983992
@@ -2,8 +2,9 @@
 Celery configuration for VoyageurCompass project.
 """
 
+import logging
 import os
-import logging
+
 from celery import Celery
 from celery.schedules import crontab
 
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Data/market_views.py:before	2025-08-12 01:15:57.045279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Data/market_views.py:after	2025-08-12 01:16:01.992139
@@ -2,18 +2,19 @@
 Additional API views for market data and synchronization.
 """
 
+import logging
+from datetime import datetime, timedelta
+
+from django.core.cache import cache
+from django.utils import timezone
 from rest_framework import status
 from rest_framework.decorators import api_view, permission_classes
-from rest_framework.permissions import IsAuthenticated, AllowAny, IsAdminUser
+from rest_framework.permissions import AllowAny, IsAdminUser, IsAuthenticated
 from rest_framework.response import Response
-from django.core.cache import cache
-from django.utils import timezone
-from datetime import datetime, timedelta
-import logging
 
 from Data.models import Stock, StockPrice
+from Data.serializers import StockPriceSerializer, StockSerializer
 from Data.services.yahoo_finance import yahoo_finance_service
-from Data.serializers import StockSerializer, StockPriceSerializer
 
 logger = logging.getLogger(__name__)
 
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Data/managers.py:before	2025-08-12 01:15:57.045279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Data/managers.py:after	2025-08-12 01:16:01.993670
@@ -3,7 +3,6 @@
 """
 
 from django.db import models
-
 
 # Data source constants to avoid circular imports
 # These match the values in DataSourceChoices
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Data/views.py:before	2025-08-12 01:15:57.047279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Data/views.py:after	2025-08-12 01:16:02.004836
@@ -6,27 +6,23 @@
 Provides REST endpoints for stock data and portfolio management.
 """
 
-from rest_framework import viewsets, status, filters
-from rest_framework.decorators import action
-from rest_framework.response import Response
-from rest_framework.permissions import IsAuthenticated, AllowAny
-from django.shortcuts import get_object_or_404
+from datetime import datetime, timedelta
+
 from django.db import models
 from django.db.models import Q
-from datetime import datetime, timedelta
+from django.shortcuts import get_object_or_404
 from django.utils import timezone
-
-from Data.models import Stock, StockPrice, Portfolio, PortfolioHolding
-from Data.serializers import (
-    StockSerializer,
-    StockDetailSerializer,
-    StockPriceSerializer,
-    PortfolioSerializer,
-    PortfolioDetailSerializer,
-    PortfolioHoldingSerializer,
-    StockSearchSerializer,
-    MarketStatusSerializer,
-)
+from rest_framework import filters, status, viewsets
+from rest_framework.decorators import action
+from rest_framework.permissions import AllowAny, IsAuthenticated
+from rest_framework.response import Response
+
+from Data.models import Portfolio, PortfolioHolding, Stock, StockPrice
+from Data.serializers import (MarketStatusSerializer,
+                              PortfolioDetailSerializer,
+                              PortfolioHoldingSerializer, PortfolioSerializer,
+                              StockDetailSerializer, StockPriceSerializer,
+                              StockSearchSerializer, StockSerializer)
 from Data.services.yahoo_finance import yahoo_finance_service
 
 
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Data/urls.py:before	2025-08-12 01:15:57.047279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Data/urls.py:after	2025-08-12 01:16:02.007731
@@ -2,22 +2,14 @@
 URL configuration for Data app.
 """
 
-from django.urls import path, include
+from django.urls import include, path
 from rest_framework.routers import DefaultRouter
-from Data.views import (
-    StockViewSet,
-    StockPriceViewSet,
-    PortfolioViewSet,
-    PortfolioHoldingViewSet,
-)
-from Data.market_views import (
-    market_overview,
-    sync_watchlist,
-    sector_performance,
-    compare_stocks,
-    economic_calendar,
-    bulk_price_update,
-)
+
+from Data.market_views import (bulk_price_update, compare_stocks,
+                               economic_calendar, market_overview,
+                               sector_performance, sync_watchlist)
+from Data.views import (PortfolioHoldingViewSet, PortfolioViewSet,
+                        StockPriceViewSet, StockViewSet)
 
 app_name = "data"
 
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Data/models.py:before	2025-08-12 01:15:57.045279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Data/models.py:after	2025-08-12 01:16:02.018586
@@ -1,13 +1,13 @@
+from datetime import datetime, timedelta
+from decimal import Decimal
+
+from dateutil.relativedelta import relativedelta
+from django.conf import settings
+from django.core.validators import MinValueValidator
 from django.db import models
-from django.core.validators import MinValueValidator
 from django.utils import timezone
-from django.conf import settings
-from datetime import timedelta
-from datetime import datetime
-from decimal import Decimal
-from dateutil.relativedelta import relativedelta
-
-from .managers import StockManager, RealDataManager
+
+from .managers import RealDataManager, StockManager
 
 
 class DataSourceChoices(models.TextChoices):
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Data/serializers.py:before	2025-08-12 01:15:57.046279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Data/serializers.py:after	2025-08-12 01:16:02.021717
@@ -3,9 +3,10 @@
 Handles data serialization/deserialization for API endpoints.
 """
 
+from django.contrib.auth.models import User
 from rest_framework import serializers
-from django.contrib.auth.models import User
-from Data.models import Stock, StockPrice, Portfolio, PortfolioHolding
+
+from Data.models import Portfolio, PortfolioHolding, Stock, StockPrice
 
 
 class StockPriceSerializer(serializers.ModelSerializer):
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Data/repo/price_reader.py:before	2025-08-12 01:15:57.046279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Data/repo/price_reader.py:after	2025-08-12 01:16:02.025332
@@ -3,20 +3,15 @@
 Provides typed interfaces for accessing historical price data required by analytics engine.
 """
 
-from datetime import datetime, date, timedelta
+from datetime import date, datetime, timedelta
 from decimal import Decimal
-from typing import List, Dict, Optional, Tuple, NamedTuple, Any
-from django.db.models import QuerySet, Min, Max, Count
+from typing import Any, Dict, List, NamedTuple, Optional, Tuple
+
+from django.db.models import Count, Max, Min, QuerySet
 from django.utils import timezone
 
-from Data.models import (
-    Stock,
-    StockPrice,
-    DataSector,
-    DataSectorPrice,
-    DataIndustry,
-    DataIndustryPrice,
-)
+from Data.models import (DataIndustry, DataIndustryPrice, DataSector,
+                         DataSectorPrice, Stock, StockPrice)
 
 
 class PriceData(NamedTuple):
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Data/repo/analytics_writer.py:before	2025-08-12 01:15:57.045279
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Data/tests/test_models.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Data/tests/test_views.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Data/tests/test_managers.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Data/services/yahoo_finance.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Data/services/data_processor.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Data/services/provider.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Data/services/synchronizer.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Data/services/tasks.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Data/management/commands/pull_aapl_3y.py Imports are incorrectly sorted and/or formatted.
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Data/repo/analytics_writer.py:after	2025-08-12 01:16:02.028484
@@ -5,12 +5,13 @@
 
 from datetime import datetime
 from decimal import Decimal
-from typing import Dict, Optional, Any
+from typing import Any, Dict, Optional
+
 from django.db import transaction
-from django.db.models import Count, Min, Max, Avg
+from django.db.models import Avg, Count, Max, Min
 from django.utils import timezone
 
-from Data.models import Stock, AnalyticsResults
+from Data.models import AnalyticsResults, Stock
 
 
 class AnalyticsWriter:
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Data/tests/test_models.py:before	2025-08-12 01:15:57.047279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Data/tests/test_models.py:after	2025-08-12 01:16:02.033664
@@ -1,16 +1,17 @@
 """
 Comprehensive tests for Data app models.
 """
+
+from datetime import date, timedelta
+from decimal import Decimal
 
 import pytest
 from django.contrib.auth.models import User
 from django.core.exceptions import ValidationError
 from django.db import IntegrityError
-from datetime import date, timedelta
-from decimal import Decimal
 from django.utils import timezone
 
-from Data.models import Stock, StockPrice, Portfolio, PortfolioHolding
+from Data.models import Portfolio, PortfolioHolding, Stock, StockPrice
 
 
 @pytest.mark.django_db
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Data/tests/test_views.py:before	2025-08-12 01:15:57.047279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Data/tests/test_views.py:after	2025-08-12 01:16:02.040378
@@ -1,17 +1,18 @@
 """
 Comprehensive tests for Data app API views.
 """
+
+from datetime import date, datetime, timedelta
+from decimal import Decimal
+from unittest.mock import patch
 
 import pytest
 from django.contrib.auth.models import User
 from django.urls import reverse
 from rest_framework import status
-from rest_framework.test import APITestCase, APIClient
-from unittest.mock import patch
-from datetime import datetime, timedelta, date
-from decimal import Decimal
-
-from Data.models import Stock, StockPrice, Portfolio, PortfolioHolding
+from rest_framework.test import APIClient, APITestCase
+
+from Data.models import Portfolio, PortfolioHolding, Stock, StockPrice
 
 
 class StockViewSetTestCase(APITestCase):
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Data/tests/test_managers.py:before	2025-08-12 01:15:57.047279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Data/tests/test_managers.py:after	2025-08-12 01:16:02.044229
@@ -6,10 +6,11 @@
 """
 
 import pytest
+from django.db import models
 from django.test import TestCase
-from django.db import models
-from Data.models import Stock, DataSourceChoices
+
 from Data.managers import DATA_SOURCE_MOCK, DATA_SOURCE_YAHOO
+from Data.models import DataSourceChoices, Stock
 
 
 class TestStockQuerySet(TestCase):
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Data/services/yahoo_finance.py:before	2025-08-12 01:15:57.046279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Data/services/yahoo_finance.py:after	2025-08-12 01:16:02.070069
@@ -4,25 +4,25 @@
 This module acts as the main interface for Yahoo Finance operations.
 """
 
+import hashlib
 import logging
+import os
+import random
 import re
+import threading
 import time
-import random
-import os
-import hashlib
-import threading
-from typing import Dict, List, Optional, Tuple
-import yfinance as yf
-import pandas as pd
-from typing import Dict, List, Optional, Any, Tuple
-from datetime import datetime, timedelta, timezone as dt_timezone
-from decimal import Decimal
 from collections import defaultdict
 from concurrent.futures import ThreadPoolExecutor, as_completed
-
+from datetime import datetime, timedelta
+from datetime import timezone as dt_timezone
+from decimal import Decimal
+from typing import Any, Dict, List, Optional, Tuple
+
+import pandas as pd
 # Configure yfinance with proper headers to handle consent pages
 # SSL verification remains enabled for security
 import requests
+import yfinance as yf
 from requests.adapters import HTTPAdapter
 from urllib3.util.retry import Retry
 
@@ -30,18 +30,13 @@
 # Keep a module-level default to pass explicitly to HTTP calls.
 DEFAULT_TIMEOUT = 30
 
+from django.db import models, transaction
+from django.utils import timezone
+
+from Data.models import (DataIndustry, DataIndustryPrice, DataSector,
+                         DataSectorPrice, Stock, StockPrice)
 from Data.services.provider import data_provider
 from Data.services.synchronizer import data_synchronizer
-from django.db import models, transaction
-from django.utils import timezone
-from Data.models import (
-    Stock,
-    StockPrice,
-    DataSector,
-    DataIndustry,
-    DataSectorPrice,
-    DataIndustryPrice,
-)
 
 logger = logging.getLogger(__name__)
 
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Data/services/data_processor.py:before	2025-08-12 01:15:57.046279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Data/services/data_processor.py:after	2025-08-12 01:16:02.077272
@@ -3,11 +3,11 @@
 Handles data transformation and processing for VoyageurCompass.
 """
 
+import json
 import logging
-from typing import Dict, List, Optional, Any
 from datetime import datetime, timedelta
 from decimal import Decimal
-import json
+from typing import Any, Dict, List, Optional
 
 logger = logging.getLogger(__name__)
 
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Data/services/provider.py:before	2025-08-12 01:15:57.046279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Data/services/provider.py:after	2025-08-12 01:16:02.081675
@@ -5,10 +5,11 @@
 """
 
 import logging
+import random
 import time
-import random
-from typing import Dict, List, Optional, Any
 from datetime import datetime, timedelta
+from typing import Any, Dict, List, Optional
+
 import yfinance as yf
 
 logger = logging.getLogger(__name__)
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Data/services/synchronizer.py:before	2025-08-12 01:15:57.046279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Data/services/synchronizer.py:after	2025-08-12 01:16:02.087130
@@ -5,12 +5,13 @@
 """
 
 import logging
-from typing import Dict, List, Optional, Tuple
 from datetime import datetime, timedelta
 from decimal import Decimal
+from typing import Dict, List, Optional, Tuple
+
+from django.core.cache import cache
 from django.db import transaction
 from django.utils import timezone
-from django.core.cache import cache
 
 from Data.models import Stock, StockPrice
 from Data.services.provider import data_provider
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Data/services/tasks.py:before	2025-08-12 01:15:57.046279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Data/services/tasks.py:after	2025-08-12 01:16:02.092238
@@ -3,13 +3,14 @@
 Handles asynchronous processing and scheduled jobs.
 """
 
+import json
+import time
+from datetime import datetime, timedelta
+
 from celery import shared_task
 from celery.utils.log import get_task_logger
 from django.core.cache import cache
 from django.utils import timezone
-from datetime import datetime, timedelta
-import json
-import time
 
 logger = get_task_logger(__name__)
 
@@ -141,8 +142,8 @@
         logger.info("Generating analytics report...")
 
         # Import models and services
-        from django.db.models import Count, Avg, Sum, Q
         from django.contrib.auth import get_user_model
+        from django.db.models import Avg, Count, Q, Sum
 
         User = get_user_model()
 
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Data/management/commands/pull_aapl_3y.py:before	2025-08-12 01:15:57.044279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Data/management/commands/pull_aapl_3y.py:after	2025-08-12 01:16:02.097338
@@ -2,13 +2,14 @@
 Management command to pull AAPL 3-year data with schema verification and database guard.
 """
 
+import logging
 import time
-import logging
 from datetime import datetime, timedelta
+
+from django.conf import settings
 from django.core.management.base import BaseCommand
 from django.db import connection
 from django.utils import timezone
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Data/management/commands/aapl_data_export.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Data/management/commands/show_evidence.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Data/management/commands/pull_sector_industry.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Data/management/commands/pull_market_data.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Data/management/commands/dbVerifySetup.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Data/management/commands/ta_bootstrap_one.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Data/migrations/0004_populate_portfolio_users.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Data/migrations/0007_mark_existing_data_as_yahoo.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Data/test/test_ta_bootstrap_one.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Data/test/test_sector_industry.py Imports are incorrectly sorted and/or formatted.
ERROR: /home/runner/work/VoyageurCompass/VoyageurCompass/Data/test/test_logging.py Imports are incorrectly sorted and/or formatted.
-from django.conf import settings
 
 from Data.services.yahoo_finance import yahoo_finance_service
 
@@ -74,7 +75,7 @@
     def verify_schema(self) -> bool:
         """Verify required fields exist in DATA tables."""
         try:
-            from Data.models import Stock, DataSectorPrice, DataIndustryPrice
+            from Data.models import DataIndustryPrice, DataSectorPrice, Stock
 
             # Check Stock model for required Stocks fields
             stock_fields = [f.name for f in Stock._meta.get_fields()]
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Data/management/commands/aapl_data_export.py:before	2025-08-12 01:15:57.044279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Data/management/commands/aapl_data_export.py:after	2025-08-12 01:16:02.100822
@@ -3,8 +3,9 @@
 """
 
 import os
+
+from django.apps import apps
 from django.core.management.base import BaseCommand
-from django.apps import apps
 from django.db import models as djm
 
 
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Data/management/commands/show_evidence.py:before	2025-08-12 01:15:57.044279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Data/management/commands/show_evidence.py:after	2025-08-12 01:16:02.102653
@@ -2,8 +2,8 @@
 Management command to show evidence with first and last 5 rows for DATA tables.
 """
 
+from django.apps import apps
 from django.core.management.base import BaseCommand
-from django.apps import apps
 from django.db import models as djm
 
 
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Data/management/commands/pull_sector_industry.py:before	2025-08-12 01:15:57.044279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Data/management/commands/pull_sector_industry.py:after	2025-08-12 01:16:02.104907
@@ -3,11 +3,13 @@
 Validates database engine and prevents SQLite usage.
 """
 
+import logging
+
 from django.core.management.base import BaseCommand, CommandError
 from django.db import connection
+
+from Data.models import DataSourceChoices, Stock
 from Data.services.yahoo_finance import yahoo_finance_service
-from Data.models import Stock, DataSourceChoices
-import logging
 
 logger = logging.getLogger(__name__)
 
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Data/management/commands/pull_market_data.py:before	2025-08-12 01:15:57.044279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Data/management/commands/pull_market_data.py:after	2025-08-12 01:16:02.107115
@@ -2,9 +2,11 @@
 Management command to pull market data from Yahoo Finance.
 """
 
+import logging
+
 from django.core.management.base import BaseCommand, CommandError
+
 from Data.services.yahoo_finance import yahoo_finance_service
-import logging
 
 logger = logging.getLogger(__name__)
 
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Data/management/commands/dbVerifySetup.py:before	2025-08-12 01:15:57.044279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Data/management/commands/dbVerifySetup.py:after	2025-08-12 01:16:02.109315
@@ -5,13 +5,9 @@
 
 from django.core.management.base import BaseCommand
 from django.db import connection
-from Data.models import (
-    Stock,
-    StockPrice,
-    Portfolio,
-    PortfolioHolding,
-    DataSourceChoices,
-)
+
+from Data.models import (DataSourceChoices, Portfolio, PortfolioHolding, Stock,
+                         StockPrice)
 
 
 class Command(BaseCommand):
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Data/management/commands/ta_bootstrap_one.py:before	2025-08-12 01:15:57.044279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Data/management/commands/ta_bootstrap_one.py:after	2025-08-12 01:16:02.115835
@@ -4,20 +4,16 @@
 Creates normalized sector/industry tables and populates EOD data.
 """
 
+import logging
+from datetime import datetime, timedelta
+
 from django.core.management.base import BaseCommand, CommandError
-from django.db import connection, transaction, models
+from django.db import connection, models, transaction
 from django.utils import timezone
-from datetime import datetime, timedelta
+
+from Data.models import (DataIndustry, DataIndustryPrice, DataSector,
+                         DataSectorPrice, Stock, StockPrice)
 from Data.services.yahoo_finance import create_yahoo_finance_service
-from Data.models import (
-    Stock,
-    StockPrice,
-    DataSector,
-    DataIndustry,
-    DataSectorPrice,
-    DataIndustryPrice,
-)
-import logging
 
 logger = logging.getLogger(__name__)
 
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Data/migrations/0004_populate_portfolio_users.py:before	2025-08-12 01:15:57.045279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Data/migrations/0004_populate_portfolio_users.py:after	2025-08-12 01:16:02.121394
@@ -2,6 +2,7 @@
 
 import secrets
 import string
+
 from django.db import migrations
 
 
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Data/migrations/0007_mark_existing_data_as_yahoo.py:before	2025-08-12 01:15:57.045279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Data/migrations/0007_mark_existing_data_as_yahoo.py:after	2025-08-12 01:16:02.125465
@@ -1,7 +1,8 @@
 # Generated migration for marking existing data as yahoo source
 
+import logging
+
 from django.db import migrations, transaction
-import logging
 
 logger = logging.getLogger(__name__)
 
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Data/test/test_ta_bootstrap_one.py:before	2025-08-12 01:15:57.047279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Data/test/test_ta_bootstrap_one.py:after	2025-08-12 01:16:02.137951
@@ -4,26 +4,20 @@
 """
 
 import unittest
-from unittest.mock import patch, MagicMock
+from datetime import date, datetime
 from decimal import Decimal
-from datetime import datetime, date
-from django.test import TestCase, override_settings
+from unittest.mock import MagicMock, patch
+
 from django.core.management import call_command
 from django.core.management.base import CommandError
 from django.db import connection
+from django.test import TestCase, override_settings
 from django.utils import timezone
 
-from Data.models import (
-    Stock,
-    StockPrice,
-    DataSector,
-    DataIndustry,
-    DataSectorPrice,
-    DataIndustryPrice,
-    DataSourceChoices,
-)
+from Data.management.commands.ta_bootstrap_one import Command
+from Data.models import (DataIndustry, DataIndustryPrice, DataSector,
+                         DataSectorPrice, DataSourceChoices, Stock, StockPrice)
 from Data.services.yahoo_finance import YahooFinanceService
-from Data.management.commands.ta_bootstrap_one import Command
 
 
 class TaBootstrapEngineGuardTest(TestCase):
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Data/test/test_sector_industry.py:before	2025-08-12 01:15:57.046279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Data/test/test_sector_industry.py:after	2025-08-12 01:16:02.142148
@@ -2,14 +2,15 @@
 Tests for sector/industry data pull functionality.
 """
 
-from django.test import TestCase
+from datetime import timedelta
+from io import StringIO
+from unittest.mock import MagicMock, patch
+
 from django.core.management import call_command
 from django.core.management.base import CommandError
 from django.db import connection
+from django.test import TestCase
 from django.utils import timezone
-from datetime import timedelta
-from unittest.mock import patch, MagicMock
-from io import StringIO
 
 from Data.models import Stock
 from Data.services.yahoo_finance import yahoo_finance_service
--- /home/runner/work/VoyageurCompass/VoyageurCompass/Data/test/test_logging.py:before	2025-08-12 01:15:57.046279
+++ /home/runner/work/VoyageurCompass/VoyageurCompass/Data/test/test_logging.py:after	2025-08-12 01:16:02.144061
@@ -1,6 +1,7 @@
 import logging
+from io import StringIO
+
 from django.test import TestCase
-from io import StringIO
 
 
 class DataLoggingTestCase(TestCase):
Skipped 1 files
Error: Process completed with exit code 1.

# notify
## Notify failure:

Run echo "❌ Tests failed!"
❌ Tests failed!
Error: Process completed with exit code 1.